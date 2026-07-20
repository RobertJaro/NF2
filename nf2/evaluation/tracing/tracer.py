from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch
from tqdm import tqdm

from nf2.evaluation.tracing.config import (
    BUTCHER_TABLES,
    RKF45_COEFFICIENTS,
    RKF45_WEIGHTS_4,
    RKF45_WEIGHTS_5,
    TraceConfig,
)
from nf2.train.model import calculate_current_from_jacobian


STATUS = {
    "active": 0,
    "boundary": 1,
    "weak_field": 2,
    "non_finite": 3,
    "max_steps": 4,
    "max_length": 5,
    "invalid_start": 6,
    "step_size_underflow": 7,
}


class BatchedFieldLineTracer:
    """Vectorized field-line tracer backed directly by an NF2 output model."""

    def __init__(self, output, geometry, config: TraceConfig | dict | None = None):
        self.output = output
        self.geometry = geometry
        self.config = config if isinstance(config, TraceConfig) else TraceConfig(**(config or {}))
        self._identity = torch.eye(3, dtype=torch.float32, device=output.device)

    def trace(self, start_coords, metrics=(), q_method=None):
        start = torch.as_tensor(start_coords, dtype=torch.float32, device=self.output.device)
        if start.ndim == 0 or start.shape[-1] != 3:
            raise ValueError("start_coords must have shape (..., 3).")
        original_shape = start.shape[:-1]
        start = start.reshape(-1, 3)
        metrics = {metrics} if isinstance(metrics, str) else set(metrics)
        supported_metrics = {
            "squashing_factor", "twist_number", "twist", "fieldline_length",
            "integrated_current_density", "fieldline_geometry",
        }
        unknown_metrics = metrics - supported_metrics
        if unknown_metrics:
            raise ValueError(f"Unknown field-line metric '{sorted(unknown_metrics)[0]}'.")
        compute_twist = "twist_number" in metrics or "twist" in metrics
        compute_current = "integrated_current_density" in metrics
        compute_q = "squashing_factor" in metrics
        q_method = self.config.q_method if q_method is None else q_method
        if compute_q and q_method not in {"tangent", "perturbed"}:
            raise ValueError("q_method must be 'tangent' or 'perturbed'.")

        # The persistent work queue keeps these slots occupied by refilling
        # them as soon as shorter field lines terminate.
        result = self._trace_seeds(start, compute_twist, compute_current, compute_q, q_method)
        result["seed_shape"] = original_shape
        result["trace_config"] = asdict(self.config)
        result["status_codes"] = dict(STATUS)
        result["boundary_names"] = dict(self.geometry.boundary_names)
        return self._reshape_result(result, original_shape)

    def _trace_seeds(self, start, compute_twist, compute_current, compute_q, q_method):
        need_jacobian = compute_twist or compute_current or (compute_q and q_method == "tangent")
        backward, forward = self._trace_bidirectional_queued(
            start, need_jacobian, compute_twist, compute_current,
            compute_q and q_method == "tangent", progress_desc="Trace field lines",
            record_paths=self.config.store_path,
        )

        valid = (forward["status"] == STATUS["boundary"]) & (backward["status"] == STATUS["boundary"])
        open_, closed = self.geometry.classify(backward["boundary_id"], forward["boundary_id"], valid)
        endpoint_delta = forward["endpoint"] - backward["endpoint"]
        result = {
            "start_coords": start,
            "forward_endpoint": forward["endpoint"],
            "backward_endpoint": backward["endpoint"],
            "forward_boundary": forward["boundary_id"],
            "backward_boundary": backward["boundary_id"],
            "forward_status": forward["status"],
            "backward_status": backward["status"],
            "fieldline_length": forward["length"] + backward["length"],
            "open": open_,
            "closed": closed,
            "footpoint_separation": torch.linalg.vector_norm(endpoint_delta, dim=-1),
            "apex": torch.maximum(forward["apex"], backward["apex"]),
        }
        if compute_twist:
            result["twist_number"] = forward["twist"] + backward["twist"]
        if compute_current:
            result["integrated_current_density"] = forward["current"] + backward["current"]

        result["open_polarity"] = self._open_polarity(result, valid)
        if compute_q:
            if start.shape[0] == 0:
                q = self._empty_q_result(start)
            elif q_method == "tangent":
                q = self._tangent_q(start, backward, forward, valid)
            elif q_method == "perturbed":
                q = self._perturbed_q(start, backward, forward, valid)
            result.update(q)
        if self.config.store_path:
            result["forward_path"] = forward["path"]
            result["backward_path"] = backward["path"]
            result["path"] = self._combine_paths(backward["path"], forward["path"])
        return result

    def _trace_bidirectional_queued(self, start, need_jacobian, compute_twist, compute_current, tangent,
                                    progress_desc="Trace field lines", record_paths=False):
        count = start.shape[0]
        work_start = start.repeat_interleave(2, dim=0)
        directions = torch.tensor([-1.0, 1.0], dtype=start.dtype, device=start.device).repeat(count)
        seed_ids = torch.arange(count, dtype=torch.long, device=start.device).repeat_interleave(2)
        work = self._trace_work_queue(
            work_start,
            directions,
            need_jacobian,
            compute_twist,
            compute_current,
            tangent,
            progress_seed_ids=seed_ids,
            progress_seed_count=count,
            progress_completion_target=2,
            progress_desc=progress_desc,
            record_paths=record_paths,
            slot_capacity=2 * self.config.batch_size,
        )

        def select(offset, direction):
            selected = {}
            for key, value in work.items():
                if key == "path":
                    selected[key] = value[:, offset::2]
                else:
                    selected[key] = value[offset::2] if torch.is_tensor(value) else value
            selected["direction"] = direction
            return selected

        return select(0, -1.0), select(1, 1.0)

    def _trace_work_queue(self, start, direction, need_jacobian, compute_twist, compute_current, tangent,
                          progress_seed_ids=None, progress_seed_count=None, progress_completion_target=1,
                          progress_desc=None, record_paths=False, slot_capacity=None):
        """Trace half-lines with persistent slots refilled from a global seed queue."""
        total = start.shape[0]
        if total == 0:
            return self._empty_direction_result(start, direction, record_paths)
        capacity = min(self.config.batch_size if slot_capacity is None else slot_capacity, total)
        device = start.device
        dtype = start.dtype
        identity = self._identity.to(device=device, dtype=dtype)

        # Global result storage is indexed by the original seed id. Only the
        # live integration state below scales with ``capacity``.
        endpoint_out = torch.empty_like(start)
        status_out = torch.full((total,), STATUS["invalid_start"], dtype=torch.long, device=device)
        boundary_out = torch.full((total,), -1, dtype=torch.long, device=device)
        length_out = torch.zeros(total, dtype=dtype, device=device)
        twist_out = torch.zeros(total, dtype=dtype, device=device)
        current_out = torch.zeros((total, 3), dtype=dtype, device=device)
        apex_out = torch.zeros(total, dtype=dtype, device=device)
        deformation_out = identity.expand(total, -1, -1).clone()

        slot_seed = torch.arange(capacity, dtype=torch.long, device=device)
        direction_values = torch.as_tensor(direction, dtype=dtype, device=device)
        if direction_values.ndim == 0:
            direction_values = direction_values.expand(total)
        else:
            direction_values = direction_values.reshape(total)
        slot_direction = direction_values[:capacity].clone()
        slot_step_size = torch.full((capacity,), self.config.step_size, dtype=dtype, device=device)
        progress_bar = None
        progress_counts = None
        if self.config.progress:
            if progress_seed_ids is None:
                progress_seed_ids = torch.arange(total, dtype=torch.long, device=device)
            if progress_seed_count is None:
                progress_seed_count = total
            progress_counts = torch.zeros(progress_seed_count, dtype=torch.long, device=device)
        paths = [[start[index].detach().cpu()] for index in range(total)] if record_paths else None
        next_seed = capacity
        occupied = torch.ones(capacity, dtype=torch.bool, device=device)
        coords = start[:capacity].clone()
        endpoint = coords.clone()
        status = torch.full((capacity,), STATUS["active"], dtype=torch.long, device=device)
        boundary_id = torch.full((capacity,), -1, dtype=torch.long, device=device)
        length = torch.zeros(capacity, dtype=dtype, device=device)
        twist = torch.zeros(capacity, dtype=dtype, device=device)
        current = torch.zeros((capacity, 3), dtype=dtype, device=device)
        apex = self.geometry.apex_coordinate(coords).to(dtype)
        deformation = identity.expand(capacity, -1, -1).clone()
        steps = torch.zeros(capacity, dtype=torch.long, device=device)

        # Persistent scratch space for the hot loop. Dynamic views are still
        # used for active lines, but full-capacity state/mask buffers are not
        # reconstructed at every integration step.
        finished = torch.zeros(capacity, dtype=torch.bool, device=device)
        boundary_fraction_buffer = torch.empty(capacity, dtype=dtype, device=device)
        length_fraction_buffer = torch.empty(capacity, dtype=dtype, device=device)
        intersections_buffer = torch.empty((capacity, 3), dtype=dtype, device=device)
        crossed_ids_buffer = torch.empty(capacity, dtype=torch.long, device=device)
        if self.config.progress:
            if progress_desc is None:
                direction_desc = "forward" if bool((direction_values > 0).all()) else "backward"
                progress_desc = f"Trace {direction_desc}"
            progress_bar = tqdm(total=progress_seed_count, desc=progress_desc, unit="lines", leave=False)

        def commit_and_refill(done_slots):
            nonlocal next_seed
            if done_slots.numel() == 0:
                return
            seed_ids = slot_seed[done_slots]
            endpoint_out[seed_ids] = endpoint[done_slots]
            status_out[seed_ids] = status[done_slots]
            boundary_out[seed_ids] = boundary_id[done_slots]
            length_out[seed_ids] = length[done_slots]
            twist_out[seed_ids] = twist[done_slots]
            current_out[seed_ids] = current[done_slots]
            apex_out[seed_ids] = apex[done_slots]
            deformation_out[seed_ids] = deformation[done_slots]
            if progress_bar is not None:
                completed_seed_ids = progress_seed_ids[seed_ids]
                unique_seed_ids = torch.unique(completed_seed_ids)
                before = progress_counts[unique_seed_ids].clone()
                progress_counts.index_add_(
                    0, completed_seed_ids, torch.ones_like(completed_seed_ids, dtype=progress_counts.dtype)
                )
                completed = (before < progress_completion_target) & (
                    progress_counts[unique_seed_ids] >= progress_completion_target
                )
                progress_bar.update(int(completed.sum().item()))

            available = total - next_seed
            refill_count = min(done_slots.numel(), available)
            refill_slots = done_slots[:refill_count]
            if refill_count:
                new_ids = torch.arange(next_seed, next_seed + refill_count, dtype=torch.long, device=device)
                next_seed += refill_count
                slot_seed[refill_slots] = new_ids
                slot_direction[refill_slots] = direction_values[new_ids]
                slot_step_size[refill_slots] = self.config.step_size
                coords[refill_slots] = start[new_ids]
                endpoint[refill_slots] = start[new_ids]
                status[refill_slots] = STATUS["active"]
                boundary_id[refill_slots] = -1
                length[refill_slots] = 0
                twist[refill_slots] = 0
                current[refill_slots] = 0
                apex[refill_slots] = self.geometry.apex_coordinate(start[new_ids]).to(dtype)
                deformation[refill_slots] = identity
                steps[refill_slots] = 0
                occupied[refill_slots] = True
            drained_slots = done_slots[refill_count:]
            if drained_slots.numel():
                occupied[drained_slots] = False
                slot_seed[drained_slots] = -1

        def retire_invalid_starts():
            # Refill repeatedly in case several consecutive queued seeds lie
            # outside the configured tracing domain.
            while occupied.any():
                invalid = occupied & ~self.geometry.contains(coords, self.config.boundary_tolerance)
                invalid_slots = torch.where(invalid)[0]
                if invalid_slots.numel() == 0:
                    break
                status[invalid_slots] = STATUS["invalid_start"]
                endpoint[invalid_slots] = coords[invalid_slots]
                commit_and_refill(invalid_slots)

        try:
            retire_invalid_starts()
            while occupied.any():
                slots = torch.where(occupied)[0]
                previous = coords[slots]
                previous_deformation = deformation[slots]
                step_result = self._rk_step(
                    previous, previous_deformation, slot_direction[slots], slot_step_size[slots],
                    need_jacobian, compute_twist, compute_current, tangent,
                )
                candidate = step_result["coords"]
                finite = torch.isfinite(candidate).all(-1) & step_result["error_finite"]
                field_ok = step_result["min_field"] >= self.config.min_field_strength
                valid_attempt = finite & field_ok
                accepted = valid_attempt & step_result["accepted"]
                slot_step_size[slots[valid_attempt]] = step_result["next_step_size"][valid_attempt]
                finished.zero_()

                underflow_slots = slots[valid_attempt & step_result["step_size_underflow"]]
                if underflow_slots.numel():
                    status[underflow_slots] = STATUS["step_size_underflow"]
                    endpoint[underflow_slots] = previous[valid_attempt & step_result["step_size_underflow"]]
                    finished[underflow_slots] = True

                bad_slots = slots[~valid_attempt]
                if bad_slots.numel():
                    non_finite = ~finite[~valid_attempt]
                    status[bad_slots] = torch.where(
                        non_finite,
                        torch.full_like(bad_slots, STATUS["non_finite"]),
                        torch.full_like(bad_slots, STATUS["weak_field"]),
                    )
                    endpoint[bad_slots] = previous[~valid_attempt]
                    finished[bad_slots] = True

                good_slots = slots[accepted]
                if good_slots.numel():
                    previous_good = previous[accepted]
                    candidate_good = candidate[accepted]
                    accepted_step = step_result["step_size"][accepted]
                    inside = self.geometry.contains(candidate_good, 0.0)
                    count_good = good_slots.numel()
                    boundary_fraction = boundary_fraction_buffer[:count_good]
                    length_fraction = length_fraction_buffer[:count_good]
                    intersections = intersections_buffer[:count_good]
                    crossed_ids = crossed_ids_buffer[:count_good]
                    boundary_fraction.fill_(float("inf"))
                    length_fraction.fill_(float("inf"))
                    intersections.copy_(candidate_good)
                    crossed_ids.fill_(-1)
                    outside = ~inside
                    if outside.any():
                        intersection, crossed_id = self.geometry.intersect_segment(
                            previous_good[outside], candidate_good[outside]
                        )
                        intersections[outside] = intersection
                        crossed_ids[outside] = crossed_id
                        boundary_fraction[outside] = (
                            torch.linalg.vector_norm(intersection - previous_good[outside], dim=-1)
                            / accepted_step[outside]
                        ).clamp(0, 1)

                    if self.config.max_length is not None:
                        length_fraction.copy_((
                            (self.config.max_length - length[good_slots]) / accepted_step
                        ).clamp_min(0))

                    # Stop at whichever event occurs first within this step;
                    # a boundary wins an exact tie.
                    boundary_event = outside & (boundary_fraction <= length_fraction)
                    length_event = (length_fraction < boundary_fraction) & (length_fraction <= 1)
                    terminating = boundary_event | length_event
                    event_fraction = torch.where(boundary_event, boundary_fraction, length_fraction)

                    if terminating.any():
                        stopped = good_slots[terminating]
                        fraction = event_fraction[terminating]
                        event_coords = previous_good[terminating] + fraction.to(dtype)[:, None] * (
                            candidate_good[terminating] - previous_good[terminating]
                        )
                        event_coords = torch.where(
                            boundary_event[terminating, None], intersections[terminating], event_coords
                        )
                        coords[stopped] = event_coords
                        endpoint[stopped] = event_coords
                        length[stopped] += accepted_step[terminating] * fraction
                        steps[stopped] += 1
                        apex[stopped] = torch.maximum(
                            apex[stopped], self.geometry.apex_coordinate(event_coords).to(dtype)
                        )
                        if compute_twist:
                            twist[stopped] += step_result["twist"][accepted][terminating] * fraction
                        if compute_current:
                            current[stopped] += (
                                step_result["current"][accepted][terminating] * fraction[:, None]
                            )
                        if tangent:
                            deformation[stopped] = previous_deformation[accepted][terminating] + fraction.to(dtype)[:, None, None] * (
                                step_result["deformation"][accepted][terminating]
                                - previous_deformation[accepted][terminating]
                            )
                        boundary_stopped = good_slots[boundary_event]
                        boundary_id[boundary_stopped] = crossed_ids[boundary_event]
                        status[boundary_stopped] = STATUS["boundary"]
                        status[good_slots[length_event]] = STATUS["max_length"]
                        finished[stopped] = True

                    continuing = ~terminating
                    continuing_slots = good_slots[continuing]
                    if continuing_slots.numel():
                        coords[continuing_slots] = candidate_good[continuing]
                        endpoint[continuing_slots] = candidate_good[continuing]
                        length[continuing_slots] += accepted_step[continuing]
                        steps[continuing_slots] += 1
                        apex[continuing_slots] = torch.maximum(
                            apex[continuing_slots],
                            self.geometry.apex_coordinate(candidate_good[continuing]).to(dtype),
                        )
                        if compute_twist:
                            twist[continuing_slots] += step_result["twist"][accepted][continuing]
                        if compute_current:
                            current[continuing_slots] += step_result["current"][accepted][continuing]
                        if tangent:
                            deformation[continuing_slots] = step_result["deformation"][accepted][continuing]

                        reached_steps = steps[continuing_slots] >= self.config.max_steps
                        if reached_steps.any():
                            stopped = continuing_slots[reached_steps]
                            status[stopped] = STATUS["max_steps"]
                            finished[stopped] = True

                if paths is not None:
                    step_points = endpoint[slots].detach().cpu()
                    for work_id, point in zip(slot_seed[slots].detach().cpu().tolist(), step_points):
                        paths[work_id].append(point)

                commit_and_refill(torch.where(finished)[0])
                retire_invalid_starts()
        finally:
            if progress_bar is not None:
                progress_bar.close()

        result = {
            "endpoint": endpoint_out,
            "status": status_out,
            "boundary_id": boundary_out,
            "length": length_out,
            "twist": twist_out,
            "current": current_out,
            "apex": apex_out,
            "deformation": deformation_out,
            "direction": direction_values,
        }
        if paths is not None:
            max_path_length = max(len(path) for path in paths)
            padded_paths = torch.full((max_path_length, total, 3), torch.nan, dtype=dtype)
            for work_id, path in enumerate(paths):
                padded_paths[:len(path), work_id] = torch.stack(path)
            result["path"] = padded_paths
        return result

    @staticmethod
    def _empty_direction_result(start, direction, record_paths=False):
        direction = torch.as_tensor(direction, dtype=start.dtype, device=start.device)
        if direction.ndim == 0:
            direction = direction.expand(0)
        result = {
            "endpoint": start.clone(),
            "status": torch.empty(0, dtype=torch.long, device=start.device),
            "boundary_id": torch.empty(0, dtype=torch.long, device=start.device),
            "length": torch.empty(0, dtype=start.dtype, device=start.device),
            "twist": torch.empty(0, dtype=start.dtype, device=start.device),
            "current": torch.empty((0, 3), dtype=start.dtype, device=start.device),
            "apex": torch.empty(0, dtype=start.dtype, device=start.device),
            "deformation": torch.empty((0, 3, 3), dtype=start.dtype, device=start.device),
            "direction": direction,
        }
        if record_paths:
            result["path"] = torch.empty((0, 0, 3), dtype=start.dtype)
        return result

    @staticmethod
    def _empty_q_result(start):
        empty = torch.empty(0, dtype=start.dtype, device=start.device)
        return {
            "squashing_factor": empty,
            "log10_q": empty.clone(),
            "q_valid": torch.empty(0, dtype=torch.bool, device=start.device),
            "q_condition_number": empty.clone(),
        }

    def _rk_step(self, coords, deformation, direction, step_size, need_jacobian, compute_twist, compute_current,
                 tangent):
        adaptive = self.config.method == "rkf45"
        if adaptive:
            coefficients = RKF45_COEFFICIENTS
            weights = RKF45_WEIGHTS_5
        else:
            coefficients, weights = BUTCHER_TABLES[self.config.method]
        direction = torch.as_tensor(direction, dtype=coords.dtype, device=coords.device)
        coordinate_direction = direction if direction.ndim == 0 else direction[:, None]
        deformation_direction = direction if direction.ndim == 0 else direction[:, None, None]
        step_size = torch.as_tensor(step_size, dtype=coords.dtype, device=coords.device).reshape(-1)
        coordinate_step = step_size[:, None]
        deformation_step = step_size[:, None, None]
        k_coords = []
        k_deformation = []
        stage_twist = []
        stage_current = []
        min_field = torch.full((coords.shape[0],), float("inf"), dtype=coords.dtype, device=coords.device)

        for stage, weight in enumerate(weights):
            stage_coords = coords
            stage_deformation = deformation
            if stage > 0:
                row = coefficients[stage - 1]
                stage_coords = coords + coordinate_step * sum(a * k for a, k in zip(row, k_coords))
                if tangent:
                    stage_deformation = deformation + deformation_step * sum(
                        a * k for a, k in zip(row, k_deformation)
                    )
            # A single model call supplies B and, when requested, its Jacobian.
            # Twist, current, and tangent-Q all reuse that same stage Jacobian.
            sampled = self.output._sample_tensor(stage_coords, compute_jacobian=need_jacobian)
            b = sampled["b"]
            norm = torch.linalg.vector_norm(b, dim=-1).clamp_min(1e-20)
            min_field = torch.minimum(min_field, norm)
            b_hat = b / norm[:, None]
            k_coords.append(coordinate_direction * b_hat)

            if need_jacobian:
                jacobian = sampled["jac_matrix"]
                if compute_twist or compute_current:
                    curl_b = calculate_current_from_jacobian(jacobian)
                    if compute_twist:
                        alpha = (curl_b * b).sum(-1) / (b.square().sum(-1).clamp_min(1e-20))
                        stage_twist.append(alpha / (4 * np.pi))
                    if compute_current:
                        stage_current.append(curl_b)
                if tangent:
                    identity = self._identity.to(device=b.device, dtype=b.dtype).expand(b.shape[0], -1, -1)
                    projector = identity - b_hat[:, :, None] * b_hat[:, None, :]
                    grad_b_hat = torch.bmm(projector, jacobian) / norm[:, None, None]
                    k_deformation.append(deformation_direction * torch.bmm(grad_b_hat, stage_deformation))

        next_coords = coords + coordinate_step * sum(weight * k for weight, k in zip(weights, k_coords))
        result = {
            "coords": next_coords,
            "min_field": min_field,
            "step_size": step_size,
        }
        if compute_twist:
            result["twist"] = step_size * sum(weight * value for weight, value in zip(weights, stage_twist))
        if compute_current:
            result["current"] = coordinate_step * sum(
                weight * value for weight, value in zip(weights, stage_current)
            )
        if tangent:
            result["deformation"] = deformation + deformation_step * sum(
                weight * value for weight, value in zip(weights, k_deformation)
            )

        if not adaptive:
            result["accepted"] = torch.ones(coords.shape[0], dtype=torch.bool, device=coords.device)
            result["next_step_size"] = step_size
            result["error_finite"] = torch.ones(coords.shape[0], dtype=torch.bool, device=coords.device)
            result["step_size_underflow"] = torch.zeros(
                coords.shape[0], dtype=torch.bool, device=coords.device
            )
            return result

        fourth_coords = coords + coordinate_step * sum(
            weight * value for weight, value in zip(RKF45_WEIGHTS_4, k_coords)
        )
        error_scale = self.config.atol + self.config.rtol * torch.maximum(coords.abs(), next_coords.abs())
        error_ratio = torch.sqrt(torch.mean(((next_coords - fourth_coords) / error_scale).square(), dim=-1))

        if tangent:
            fourth_deformation = deformation + deformation_step * sum(
                weight * value for weight, value in zip(RKF45_WEIGHTS_4, k_deformation)
            )
            deformation_scale = self.config.atol + self.config.rtol * torch.maximum(
                deformation.abs(), result["deformation"].abs()
            )
            deformation_error = torch.sqrt(torch.mean(
                ((result["deformation"] - fourth_deformation) / deformation_scale).square(), dim=(-2, -1)
            ))
            error_ratio = torch.maximum(error_ratio, deformation_error)

        if compute_twist:
            fourth_twist = step_size * sum(
                weight * value for weight, value in zip(RKF45_WEIGHTS_4, stage_twist)
            )
            twist_scale = self.config.atol + self.config.rtol * torch.maximum(
                result["twist"].abs(), fourth_twist.abs()
            )
            error_ratio = torch.maximum(error_ratio, (result["twist"] - fourth_twist).abs() / twist_scale)

        if compute_current:
            fourth_current = coordinate_step * sum(
                weight * value for weight, value in zip(RKF45_WEIGHTS_4, stage_current)
            )
            current_scale = self.config.atol + self.config.rtol * torch.maximum(
                result["current"].abs(), fourth_current.abs()
            )
            current_error = torch.sqrt(torch.mean(
                ((result["current"] - fourth_current) / current_scale).square(), dim=-1
            ))
            error_ratio = torch.maximum(error_ratio, current_error)

        finite_error = torch.isfinite(error_ratio)
        at_minimum = step_size <= self.config.min_step_size * (1 + 1e-6)
        result["accepted"] = finite_error & (error_ratio <= 1)
        result["step_size_underflow"] = finite_error & (error_ratio > 1) & at_minimum
        safe_error = torch.where(finite_error, error_ratio.clamp_min(1e-20), torch.full_like(error_ratio, float("inf")))
        factor = (0.9 * safe_error.pow(-0.2)).clamp(0.2, 5.0)
        result["next_step_size"] = (step_size * factor).clamp(
            self.config.min_step_size, self.config.max_step_size
        )
        result["error_finite"] = finite_error
        return result

    def _open_polarity(self, result, valid):
        backward_inner = result["backward_boundary"] == self.geometry.inner_boundary_id
        forward_inner = result["forward_boundary"] == self.geometry.inner_boundary_id
        endpoint = torch.where(backward_inner[:, None], result["backward_endpoint"], result["forward_endpoint"])
        polarity = torch.zeros(endpoint.shape[0], dtype=torch.int8, device=endpoint.device)
        open_with_inner = result["open"] & (backward_inner | forward_inner) & valid
        if open_with_inner.any():
            sampled = self.output._sample_tensor(endpoint[open_with_inner], compute_jacobian=False)
            normal = self.geometry.inner_normal(endpoint[open_with_inner])
            polarity[open_with_inner] = torch.sign((sampled["b"] * normal).sum(-1)).to(torch.int8)
        return polarity

    def _endpoint_map(self, start, direction_result, seed_basis):
        endpoint = direction_result["endpoint"]
        boundary_id = direction_result["boundary_id"]
        endpoint_basis = self.geometry.surface_basis(endpoint, boundary_id)
        domain_normal = self.geometry.boundary_normal(endpoint, boundary_id)
        sampled = self.output._sample_tensor(endpoint, compute_jacobian=False)
        b_hat = sampled["b"] / torch.linalg.vector_norm(sampled["b"], dim=-1, keepdim=True).clamp_min(1e-20)
        flow = direction_result["direction"] * b_hat
        denominator = (domain_normal * flow).sum(-1)
        correction = torch.eye(3, dtype=start.dtype, device=start.device).expand(start.shape[0], -1, -1).clone()
        safe_denominator = torch.where(
            denominator.abs() > 1e-12, denominator, torch.ones_like(denominator)
        )
        correction -= flow[:, :, None] * domain_normal[:, None, :] / safe_denominator[:, None, None]
        mapped = torch.bmm(correction, torch.bmm(direction_result["deformation"], seed_basis))
        surface_map = torch.bmm(endpoint_basis.transpose(1, 2), mapped)
        return surface_map, denominator

    def _q_from_maps(self, backward_map, forward_map, valid):
        condition = torch.linalg.cond(backward_map)
        determinant_backward = torch.linalg.det(backward_map)
        map_valid = valid & torch.isfinite(condition) & (condition < self.config.q_condition_limit) & (
            determinant_backward.abs() > 1e-12
        )
        safe_backward = backward_map.clone()
        safe_backward[~map_valid] = torch.eye(2, dtype=safe_backward.dtype, device=safe_backward.device)
        mapping = torch.bmm(forward_map, torch.linalg.inv(safe_backward))
        determinant = torch.linalg.det(mapping)
        mapping_condition = torch.linalg.cond(mapping)
        q = mapping.square().sum((-2, -1)) / determinant.abs().clamp_min(1e-30)
        map_valid &= (
            torch.isfinite(q) & torch.isfinite(determinant) & torch.isfinite(mapping_condition)
            & (determinant.abs() > 1e-12)
        )
        q = torch.where(map_valid, q.clamp_min(2.0), torch.full_like(q, torch.nan))
        return {
            "squashing_factor": q,
            "log10_q": torch.log10(q),
            "q_valid": map_valid,
            "q_condition_number": mapping_condition,
        }

    def _tangent_q(self, start, backward, forward, valid):
        start_b = self.output._sample_tensor(start, compute_jacobian=False)["b"]
        seed_basis = self.geometry.seed_basis(start_b)
        backward_map, backward_denominator = self._endpoint_map(start, backward, seed_basis)
        forward_map, forward_denominator = self._endpoint_map(start, forward, seed_basis)
        valid = valid & (backward_denominator.abs() > 1e-8) & (forward_denominator.abs() > 1e-8)
        return self._q_from_maps(backward_map, forward_map, valid)

    def _perturbed_q(self, start, backward, forward, valid):
        epsilon = self.config.q_epsilon or self.config.step_size / 4
        b = self.output._sample_tensor(start, compute_jacobian=False)["b"]
        basis = self.geometry.seed_basis(b)
        backward_at_seed = torch.linalg.vector_norm(start - backward["endpoint"], dim=-1) <= (
            self.config.boundary_tolerance * 10
        )
        forward_at_seed = torch.linalg.vector_norm(start - forward["endpoint"], dim=-1) <= (
            self.config.boundary_tolerance * 10
        )
        if backward_at_seed.any():
            boundary_basis = self.geometry.surface_basis(backward["endpoint"], backward["boundary_id"])
            basis = torch.where(backward_at_seed[:, None, None], boundary_basis, basis)
        if forward_at_seed.any():
            boundary_basis = self.geometry.surface_basis(forward["endpoint"], forward["boundary_id"])
            basis = torch.where(forward_at_seed[:, None, None], boundary_basis, basis)
        offsets = torch.stack((
            epsilon * basis[..., 0],
            -epsilon * basis[..., 0],
            epsilon * basis[..., 1],
            -epsilon * basis[..., 1],
        ), dim=1)
        perturbed = (start[:, None, :] + offsets).reshape(-1, 3)
        minus, plus = self._trace_bidirectional_queued(
            perturbed, False, False, False, False, progress_desc="Trace Q stencil", record_paths=False
        )
        n = start.shape[0]
        plus_end = plus["endpoint"].reshape(n, 4, 3)
        minus_end = minus["endpoint"].reshape(n, 4, 3)
        plus_id = plus["boundary_id"].reshape(n, 4)
        minus_id = minus["boundary_id"].reshape(n, 4)
        plus_status = plus["status"].reshape(n, 4)
        minus_status = minus["status"].reshape(n, 4)
        stencil_valid = (
            (plus_status == STATUS["boundary"]).all(-1)
            & (minus_status == STATUS["boundary"]).all(-1)
            & (plus_id == plus_id[:, :1]).all(-1)
            & (minus_id == minus_id[:, :1]).all(-1)
        )
        plus_basis = self.geometry.surface_basis(forward["endpoint"], forward["boundary_id"])
        minus_basis = self.geometry.surface_basis(backward["endpoint"], backward["boundary_id"])
        plus_derivatives = torch.stack((plus_end[:, 0] - plus_end[:, 1], plus_end[:, 2] - plus_end[:, 3]), dim=-1) / (2 * epsilon)
        minus_derivatives = torch.stack((minus_end[:, 0] - minus_end[:, 1], minus_end[:, 2] - minus_end[:, 3]), dim=-1) / (2 * epsilon)
        plus_map = torch.bmm(plus_basis.transpose(1, 2), plus_derivatives)
        minus_map = torch.bmm(minus_basis.transpose(1, 2), minus_derivatives)
        return self._q_from_maps(minus_map, plus_map, valid & stencil_valid)

    @staticmethod
    def _combine_paths(backward_path, forward_path):
        """Combine stored half-lines into endpoint-to-endpoint padded paths."""
        combined = []
        count = backward_path.shape[1]
        for index in range(count):
            backward = backward_path[:, index]
            forward = forward_path[:, index]
            backward = backward[torch.isfinite(backward).all(-1)]
            forward = forward[torch.isfinite(forward).all(-1)]

            def remove_repeated(points):
                if points.shape[0] < 2:
                    return points
                keep = torch.ones(points.shape[0], dtype=torch.bool)
                keep[1:] = torch.linalg.vector_norm(points[1:] - points[:-1], dim=-1) > 1e-10
                return points[keep]

            backward = remove_repeated(backward)
            forward = remove_repeated(forward)
            combined.append(torch.cat((backward.flip(0), forward[1:]), dim=0))

        max_length = max((path.shape[0] for path in combined), default=0)
        padded = torch.full((max_length, count, 3), torch.nan, dtype=backward_path.dtype)
        for index, path in enumerate(combined):
            padded[:path.shape[0], index] = path
        return padded

    @staticmethod
    def _reshape_result(result, shape):
        public = {}
        for key, value in result.items():
            if key in {"seed_shape", "trace_config"}:
                public[key] = value
                continue
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
                if key == "path" or key.endswith("_path"):
                    public[key] = value.reshape((value.shape[0], *shape, 3))
                else:
                    public[key] = value.reshape((*shape, *value.shape[1:]))
            else:
                public[key] = value
        return public
