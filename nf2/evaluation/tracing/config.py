from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class TraceConfig:
    """Numerical controls for batched field-line tracing.

    Step sizes and maximum lengths are expressed in normalized NF2 model
    coordinates. ``max_length`` limits each forward/backward half-line, so a
    combined line can be up to twice that length. Public output helpers convert
    physical defaults to these units before constructing a tracer.
    """

    method: str = "rkf45"
    step_size: float = 0.01
    rtol: float = 1e-5
    atol: float = 1e-7
    min_step_size: float | None = None
    max_step_size: float | None = None
    max_steps: int = 2_000
    max_length: float | None = None
    min_field_strength: float = 1e-8
    boundary_tolerance: float = 1e-6
    batch_size: int = 65_536
    store_path: bool = False
    progress: bool = False
    q_method: str = "tangent"
    q_epsilon: float | None = None
    q_condition_limit: float = 1e8

    def __post_init__(self):
        method = self.method.lower().replace("-", "")
        aliases = {
            "rk1": "euler", "euler": "euler", "rk2": "rk2", "rk3": "rk3", "rk4": "rk4",
            "rkf45": "rkf45", "fehlberg45": "rkf45",
        }
        if method not in aliases:
            raise ValueError("Trace method must be one of: euler/rk1, rk2, rk3, rk4, rkf45.")
        object.__setattr__(self, "method", aliases[method])
        if not isfinite(self.step_size) or self.step_size <= 0:
            raise ValueError("Trace step_size must be positive.")
        if not isfinite(self.rtol) or self.rtol <= 0:
            raise ValueError("Trace rtol must be positive.")
        if not isfinite(self.atol) or self.atol <= 0:
            raise ValueError("Trace atol must be positive.")
        min_step_size = self.step_size * 1e-3 if self.min_step_size is None else self.min_step_size
        max_step_size = self.step_size * 10 if self.max_step_size is None else self.max_step_size
        if not isfinite(min_step_size) or min_step_size <= 0:
            raise ValueError("Trace min_step_size must be positive.")
        if not isfinite(max_step_size) or max_step_size < min_step_size:
            raise ValueError("Trace max_step_size must be at least min_step_size.")
        object.__setattr__(self, "min_step_size", min_step_size)
        object.__setattr__(self, "max_step_size", max_step_size)
        if self.max_steps <= 0:
            raise ValueError("Trace max_steps must be positive.")
        if self.max_length is not None and (not isfinite(self.max_length) or self.max_length <= 0):
            raise ValueError("Trace max_length must be positive when provided.")
        if self.batch_size <= 0:
            raise ValueError("Trace batch_size must be positive.")
        if not isfinite(self.min_field_strength) or self.min_field_strength < 0:
            raise ValueError("min_field_strength must be finite and non-negative.")
        if not isfinite(self.boundary_tolerance) or self.boundary_tolerance < 0:
            raise ValueError("boundary_tolerance must be finite and non-negative.")
        if self.q_method not in {"tangent", "perturbed"}:
            raise ValueError("q_method must be 'tangent' or 'perturbed'.")
        if self.q_epsilon is not None and (not isfinite(self.q_epsilon) or self.q_epsilon <= 0):
            raise ValueError("q_epsilon must be positive when provided.")
        if not isfinite(self.q_condition_limit) or self.q_condition_limit <= 0:
            raise ValueError("q_condition_limit must be finite and positive.")


BUTCHER_TABLES = {
    "euler": ((), (1.0,)),
    "rk2": (((0.5,),), (0.0, 1.0)),
    "rk3": (((0.5,), (-1.0, 2.0)), (1 / 6, 2 / 3, 1 / 6)),
    "rk4": (((0.5,), (0.0, 0.5), (0.0, 0.0, 1.0)), (1 / 6, 1 / 3, 1 / 3, 1 / 6)),
}


# Fehlberg's embedded fourth/fifth-order pair. The fifth-order solution is
# accepted; the fourth-order weights provide its local truncation estimate.
RKF45_COEFFICIENTS = (
    (1 / 4,),
    (3 / 32, 9 / 32),
    (1932 / 2197, -7200 / 2197, 7296 / 2197),
    (439 / 216, -8, 3680 / 513, -845 / 4104),
    (-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40),
)
RKF45_WEIGHTS_5 = (16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55)
RKF45_WEIGHTS_4 = (25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0)
