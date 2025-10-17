#!/usr/bin/env python3
"""Test script to validate Kitchen training configurations work correctly.

This script tests all Kitchen task and network combinations using the debug configuration
to ensure each setup runs without errors. Only tests a single seed (99).

Author: Chaoyi Pan
Date: 2025-10-17
"""

import subprocess
import sys

# ANSI color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def run_test(
    script: str, task: str, network: str, loss_type: str = "flow"
) -> tuple[bool, str]:
    """Run a single test configuration.

    Args:
        script: Training script path (e.g., "examples/train_kitchen.py")
        task: Task configuration name
        network: Network configuration name
        loss_type: Loss function type

    Returns:
        Tuple of (success: bool, error_message: str)
    """
    cmd = [
        "uv",
        "run",
        script,
        "-cn",
        "exps/debug.yaml",
        f"task={task}",
        f"network={network}",
        f"optimization.loss_type={loss_type}",
        "optimization.seed=99",
    ]

    print(
        f"  Testing: {BOLD}{task}{RESET} + {BOLD}{network}{RESET} + {BOLD}{loss_type}{RESET}...",
        end=" ",
    )

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1500,  # 25 minute timeout
        )

        if result.returncode == 0:
            print(f"{GREEN}✓ PASSED{RESET}")
            return True, ""
        else:
            error_msg = result.stderr[-500:] if result.stderr else result.stdout[-500:]
            print(f"{RED}✗ FAILED{RESET}")
            return False, error_msg

    except subprocess.TimeoutExpired:
        print(f"{RED}✗ TIMEOUT{RESET}")
        return False, "Test timed out after 25 minutes"
    except Exception as e:
        print(f"{RED}✗ ERROR{RESET}")
        return False, str(e)


def main():
    """Main test runner."""
    print(f"\n{BOLD}{'=' * 80}{RESET}")
    print(f"{BOLD}Running Kitchen Training Configuration Tests{RESET}")
    print(f"{BOLD}{'=' * 80}{RESET}\n")

    # Define test configurations
    kitchen_tasks = [
        "kitchen_state",
    ]

    networks = [
        "sudeepdit",
        "chitransformer",
        "chiunet",
        "mlp",
        "rnn",
    ]

    loss_types = ["flow", "mip", "regression"]

    # Track results
    results = {
        "passed": [],
        "failed": [],
    }

    # Test Kitchen configurations
    print(f"{YELLOW}{BOLD}Testing Kitchen Configurations{RESET}\n")
    for task in kitchen_tasks:
        print(f"{BOLD}Task: {task}{RESET}")
        for network in networks:
            for loss_type in loss_types:
                success, error = run_test(
                    "examples/train_kitchen.py",
                    task,
                    network,
                    loss_type,
                )

                test_name = f"kitchen/{task}/{network}/{loss_type}"
                if success:
                    results["passed"].append(test_name)
                else:
                    results["failed"].append((test_name, error))
        print()  # Empty line between tasks

    # Print summary
    print(f"\n{BOLD}{'=' * 80}{RESET}")
    print(f"{BOLD}Test Summary{RESET}")
    print(f"{BOLD}{'=' * 80}{RESET}\n")

    total_tests = len(results["passed"]) + len(results["failed"])
    print(f"Total tests run: {BOLD}{total_tests}{RESET}")
    print(f"{GREEN}Passed: {len(results['passed'])}{RESET}")
    print(f"{RED}Failed: {len(results['failed'])}{RESET}")

    if results["failed"]:
        print(f"\n{RED}{BOLD}Failed Tests:{RESET}")
        for test_name, error in results["failed"]:
            print(f"  {RED}✗{RESET} {test_name}")
            if error:
                # Print first line of error for quick diagnosis
                first_line = error.split("\n")[0] if error else ""
                print(f"    Error: {first_line[:100]}")

        print(
            f"\n{YELLOW}Re-run failed tests individually for full error output:{RESET}"
        )
        for test_name, _ in results["failed"]:
            parts = test_name.split("/")
            script = f"examples/train_{parts[0]}.py"
            task = parts[1]
            network = parts[2]
            loss_type = parts[3]
            print(
                f"  uv run {script} -cn exps/debug.yaml task={task} network={network} optimization.loss_type={loss_type}"
            )

        sys.exit(1)
    else:
        print(f"\n{GREEN}{BOLD}All tests passed! ✨{RESET}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
