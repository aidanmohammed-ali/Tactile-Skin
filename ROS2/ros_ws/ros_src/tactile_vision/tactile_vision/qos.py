from __future__ import annotations

from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


def image_qos_profile(depth: int = 1, reliability: str = "best_effort") -> QoSProfile:
    reliability_policy = _parse_reliability(reliability)
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=max(1, int(depth)),
        reliability=reliability_policy,
        durability=DurabilityPolicy.VOLATILE,
    )


def _parse_reliability(value: str) -> ReliabilityPolicy:
    text = str(value).strip().lower().replace("-", "_")
    if text in ("best_effort", "besteffort"):
        return ReliabilityPolicy.BEST_EFFORT
    if text == "reliable":
        return ReliabilityPolicy.RELIABLE
    raise ValueError(f"unsupported image QoS reliability: {value!r}")
