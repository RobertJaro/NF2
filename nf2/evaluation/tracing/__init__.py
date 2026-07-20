"""Tensor-native magnetic field-line tracing for NF2 output objects."""

from nf2.evaluation.tracing.config import TraceConfig
from nf2.evaluation.tracing.geometry import CartesianTraceGeometry, SphericalTraceGeometry
from nf2.evaluation.tracing.tracer import BatchedFieldLineTracer

__all__ = [
    "BatchedFieldLineTracer",
    "CartesianTraceGeometry",
    "SphericalTraceGeometry",
    "TraceConfig",
]
