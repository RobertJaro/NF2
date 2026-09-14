from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class TraceGeometry(ABC):
    """Geometry-dependent domain and endpoint operations."""

    inner_boundary_id: int
    outer_boundary_ids: tuple[int, ...]

    @abstractmethod
    def contains(self, coords: torch.Tensor, tolerance: float = 0.0) -> torch.Tensor:
        pass

    @abstractmethod
    def intersect_segment(self, start: torch.Tensor, end: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def boundary_normal(self, coords: torch.Tensor, boundary_id: torch.Tensor) -> torch.Tensor:
        """Return the outward normal of the traced domain."""

    @abstractmethod
    def inner_normal(self, coords: torch.Tensor) -> torch.Tensor:
        """Return the physically positive normal at the photospheric/inner boundary."""

    @abstractmethod
    def apex_coordinate(self, coords: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def boundary_residual(self, coords: torch.Tensor, boundary_id: torch.Tensor) -> torch.Tensor:
        """Signed distance-like residual, negative inside and zero on the selected boundary."""

    @abstractmethod
    def project_to_boundary(self, coords: torch.Tensor, boundary_id: torch.Tensor) -> torch.Tensor:
        """Project points onto the selected boundary."""

    def surface_basis(self, coords: torch.Tensor, boundary_id: torch.Tensor) -> torch.Tensor:
        normal = self.boundary_normal(coords, boundary_id)
        return perpendicular_basis(normal)

    def seed_basis(self, b: torch.Tensor) -> torch.Tensor:
        return perpendicular_basis(b)

    def classify(self, backward_id: torch.Tensor, forward_id: torch.Tensor, valid: torch.Tensor):
        backward_inner = backward_id == self.inner_boundary_id
        forward_inner = forward_id == self.inner_boundary_id
        closed = valid & backward_inner & forward_inner
        open_ = valid & ((backward_inner & ~forward_inner) | (~backward_inner & forward_inner))
        return open_, closed


def perpendicular_basis(vectors: torch.Tensor) -> torch.Tensor:
    """Build stable right-handed orthonormal tangent bases, shape (..., 3, 2)."""
    normal = vectors / torch.linalg.vector_norm(vectors, dim=-1, keepdim=True).clamp_min(1e-20)
    axes = torch.eye(3, dtype=normal.dtype, device=normal.device)
    reference = axes[torch.argmin(torch.abs(normal), dim=-1)]
    first = torch.linalg.cross(normal, reference, dim=-1)
    first = first / torch.linalg.vector_norm(first, dim=-1, keepdim=True).clamp_min(1e-20)
    second = torch.linalg.cross(normal, first, dim=-1)
    return torch.stack((first, second), dim=-1)


class CartesianTraceGeometry(TraceGeometry):
    """Axis-aligned Cartesian tracing box in normalized model coordinates."""

    inner_boundary_id = 4  # z minimum
    outer_boundary_ids = (0, 1, 2, 3, 5)
    boundary_names = {0: "x_min", 1: "x_max", 2: "y_min", 3: "y_max", 4: "z_min", 5: "z_max"}

    def __init__(self, bounds):
        self.bounds = torch.stack([torch.as_tensor(pair, dtype=torch.float32) for pair in bounds])
        if self.bounds.shape != (3, 2):
            raise ValueError("Cartesian bounds must have shape (3, 2).")

    def _bounds(self, coords):
        return self.bounds.to(device=coords.device, dtype=coords.dtype)

    def contains(self, coords, tolerance=0.0):
        bounds = self._bounds(coords)
        return ((coords >= bounds[:, 0] - tolerance) & (coords <= bounds[:, 1] + tolerance)).all(-1)

    def intersect_segment(self, start, end):
        bounds = self._bounds(start)
        delta = end - start
        n = start.shape[0]
        best_t = torch.full((n,), float("inf"), dtype=start.dtype, device=start.device)
        boundary_id = torch.full((n,), -1, dtype=torch.long, device=start.device)
        for axis in range(3):
            for side in range(2):
                denominator = delta[:, axis]
                t = (bounds[axis, side] - start[:, axis]) / torch.where(
                    denominator.abs() > 1e-20, denominator, torch.ones_like(denominator)
                )
                point = start + t[:, None] * delta
                other = [i for i in range(3) if i != axis]
                on_face = (
                    (denominator.abs() > 1e-20)
                    & (t >= -1e-4)
                    & (t <= 1 + 1e-7)
                    & (point[:, other] >= bounds[other, 0] - 1e-6).all(-1)
                    & (point[:, other] <= bounds[other, 1] + 1e-6).all(-1)
                )
                update = on_face & (t < best_t)
                best_t = torch.where(update, t, best_t)
                boundary_id = torch.where(update, torch.full_like(boundary_id, axis * 2 + side), boundary_id)
        best_t = torch.where(torch.isfinite(best_t), best_t.clamp(0, 1), torch.ones_like(best_t))
        return start + best_t[:, None] * delta, boundary_id

    def boundary_normal(self, coords, boundary_id):
        normals = torch.zeros_like(coords)
        axis = torch.div(boundary_id.clamp_min(0), 2, rounding_mode="floor")
        side = boundary_id.clamp_min(0) % 2
        normals.scatter_(1, axis[:, None], torch.where(side == 0, -1.0, 1.0)[:, None].to(coords))
        return normals

    def inner_normal(self, coords):
        normal = torch.zeros_like(coords)
        normal[:, 2] = 1
        return normal

    def apex_coordinate(self, coords):
        return coords[..., 2]

    def boundary_residual(self, coords, boundary_id):
        bounds = self._bounds(coords)
        axis = torch.div(boundary_id, 2, rounding_mode="floor")
        side = boundary_id % 2
        value = coords.gather(1, axis[:, None]).squeeze(1)
        target = bounds[axis, side]
        return torch.where(side == 0, target - value, value - target)

    def project_to_boundary(self, coords, boundary_id):
        bounds = self._bounds(coords)
        axis = torch.div(boundary_id, 2, rounding_mode="floor")
        side = boundary_id % 2
        projected = coords.clone()
        projected.scatter_(1, axis[:, None], bounds[axis, side][:, None])
        return projected


class SphericalTraceGeometry(TraceGeometry):
    """Spherical shell traced in nonsingular Cartesian model coordinates."""

    inner_boundary_id = 0
    outer_boundary_ids = (1,)
    boundary_names = {0: "inner_radius", 1: "outer_radius"}

    def __init__(self, radius_range):
        self.radius_range = tuple(float(v) for v in radius_range)
        if len(self.radius_range) != 2 or self.radius_range[1] <= self.radius_range[0]:
            raise ValueError("Spherical radius_range must contain increasing inner and outer radii.")

    def contains(self, coords, tolerance=0.0):
        radius = torch.linalg.vector_norm(coords, dim=-1)
        return (radius >= self.radius_range[0] - tolerance) & (radius <= self.radius_range[1] + tolerance)

    def intersect_segment(self, start, end):
        end_radius = torch.linalg.vector_norm(end, dim=-1)
        inner = end_radius < self.radius_range[0]
        target = torch.where(
            inner,
            torch.full_like(end_radius, self.radius_range[0]),
            torch.full_like(end_radius, self.radius_range[1]),
        )
        low = torch.zeros_like(target)
        high = torch.ones_like(target)
        for _ in range(28):
            mid = (low + high) / 2
            point = start + mid[:, None] * (end - start)
            radius = torch.linalg.vector_norm(point, dim=-1)
            still_inside = torch.where(inner, radius >= target, radius <= target)
            low = torch.where(still_inside, mid, low)
            high = torch.where(still_inside, high, mid)
        point = start + high[:, None] * (end - start)
        return point, torch.where(inner, torch.zeros_like(target, dtype=torch.long), torch.ones_like(target, dtype=torch.long))

    def boundary_normal(self, coords, boundary_id):
        radial = coords / torch.linalg.vector_norm(coords, dim=-1, keepdim=True).clamp_min(1e-20)
        return torch.where((boundary_id == self.inner_boundary_id)[:, None], -radial, radial)

    def inner_normal(self, coords):
        return coords / torch.linalg.vector_norm(coords, dim=-1, keepdim=True).clamp_min(1e-20)

    def apex_coordinate(self, coords):
        return torch.linalg.vector_norm(coords, dim=-1)

    def boundary_residual(self, coords, boundary_id):
        radius = torch.linalg.vector_norm(coords, dim=-1)
        inner = boundary_id == self.inner_boundary_id
        target = torch.where(
            inner,
            torch.full_like(radius, self.radius_range[0]),
            torch.full_like(radius, self.radius_range[1]),
        )
        return torch.where(inner, target - radius, radius - target)

    def project_to_boundary(self, coords, boundary_id):
        radius = torch.linalg.vector_norm(coords, dim=-1, keepdim=True).clamp_min(1e-20)
        inner = boundary_id == self.inner_boundary_id
        target = torch.where(
            inner,
            torch.full_like(boundary_id, self.radius_range[0], dtype=coords.dtype),
            torch.full_like(boundary_id, self.radius_range[1], dtype=coords.dtype),
        )
        return coords * (target[:, None] / radius)
