"""Tests for the WarmupAnnealingScheduler class."""

from mip.scheduler import WarmupAnnealingScheduler


def test_warmup_annealing_scheduler():
    scheduler = WarmupAnnealingScheduler(
        max_steps=1000, warmup_ratio=0.1, rampup_ratio=0.8, min_value=0.0, max_value=1.0
    )

    # Test warmup phase: steps 0-99 should return min_value (0.0)
    assert scheduler(0) == 0.0
    assert scheduler(50) == 0.0
    assert scheduler(99) == 0.0

    # Test rampup phase: steps 100-899 should linearly interpolate from 0.0 to 1.0
    assert scheduler(100) == 0.0  # Start of rampup
    assert abs(scheduler(500) - 0.5) < 1e-6  # Middle of rampup
    assert abs(scheduler(899) - (799 / 800)) < 1e-6  # Near end of rampup

    # Test post-rampup phase: steps 900+ should return max_value (1.0)
    assert scheduler(900) == 1.0
    assert scheduler(1000) == 1.0
    assert scheduler(1500) == 1.0


if __name__ == "__main__":
    test_warmup_annealing_scheduler()
