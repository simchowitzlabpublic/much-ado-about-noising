#!/usr/bin/env python
"""Test training for all robomimic task configurations."""

import subprocess
import sys

# Define all task configurations to test
TASKS = {
    # lift task
    "lift_ph": ["state", "image"],
    "lift_mh": ["state", "image"],
    # can task
    "can_ph": ["state", "image"],
    "can_mh": ["state", "image"],
    # square task
    "square_ph": ["state", "image"],
    "square_mh": ["state", "image"],
    # transport task
    "transport_ph": ["state", "image"],
    "transport_mh": ["state", "image"],
    # tool_hang task (only ph)
    "tool_hang_ph": ["state", "image"],
}

# Test parameters
TEST_STEPS = 10  # Run for only 10 steps to verify the config works
BATCH_SIZE = 4  # Small batch size for testing


def test_task(task_name, obs_type):
    """Test a single task configuration."""
    # Determine the config file name
    config_name = f"{task_name}_{obs_type}"

    print(f"\n{'=' * 60}")
    print(f"Testing: {config_name} (obs_type={obs_type})")
    print(f"{'=' * 60}")

    # Build command
    cmd = [
        "uv",
        "run",
        "python",
        "examples/train_robomimic.py",
        "-cn",
        "exps/debug.yaml",
        f"task={config_name}",
        f"optimization.gradient_steps={TEST_STEPS}",
        f"optimization.batch_size={BATCH_SIZE}",
        "log.log_freq=1",
        "log.eval_freq=10",
        "log.wandb_mode=disabled",  # Disable wandb for testing
    ]

    print(f"Command: {' '.join(cmd)}")

    try:
        # Run the training command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
            check=False,
        )

        if result.returncode == 0:
            # Check if training actually ran
            if "Step" in result.stdout and "loss" in result.stdout:
                print(f"✅ SUCCESS: {config_name} training completed")
                return True
            else:
                print(f"⚠️ WARNING: {config_name} ran but no training output detected")
                return False
        else:
            print(f"❌ FAILED: {config_name}")
            if "No such config group" in result.stderr:
                print("  Error: Config file not found")
            elif "FileNotFoundError" in result.stderr:
                print("  Error: Dataset file not found")
            elif result.stderr:
                # Print first few lines of error
                error_lines = result.stderr.split("\n")[:5]
                for line in error_lines:
                    if line.strip():
                        print(f"  {line}")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {config_name} took too long")
        return False
    except Exception as e:
        print(f"❌ ERROR: {config_name} - {e}")
        return False


def main():
    """Test all task configurations."""
    print("=" * 80)
    print("TESTING ALL ROBOMIMIC TASK CONFIGURATIONS")
    print("=" * 80)
    print(f"Tasks to test: {len(TASKS)}")
    print(f"Test steps: {TEST_STEPS}")
    print(f"Batch size: {BATCH_SIZE}")
    print("=" * 80)

    results = {}

    # Test each task configuration
    for task_name, obs_types in TASKS.items():
        for obs_type in obs_types:
            config_key = f"{task_name}_{obs_type}"
            success = test_task(task_name, obs_type)
            results[config_key] = success

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for config_key, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {config_key}")

    print(f"\n{passed}/{total} configurations passed")

    if passed == total:
        print("🎉 All configurations tested successfully!")
    else:
        print(f"⚠️ {total - passed} configurations failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
