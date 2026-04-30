"""
Current Main Diagnostic Pipeline

Runs the current mainline SVT experiments:
  1. v4.2: Conflict-first ObjectFile stress test
  2. v5.1: Scaling sanity audit

Usage:
    python scripts/run_current_main.py

Outputs:
    results/svt_v4_2_conflict_first_object_file/
    results/svt_v5_1_scaling_sanity_audit/
"""

import subprocess
import sys


def run_script(script_path):
    print(f"\n{'='*70}")
    print(f"Running: {script_path}")
    print(f"{'='*70}\n")
    result = subprocess.run([sys.executable, script_path], cwd=".")
    if result.returncode != 0:
        print(f"Warning: {script_path} exited with code {result.returncode}")
    return result.returncode


def main():
    print("SVT Current Main Diagnostic Pipeline")
    print("=" * 70)
    print("This will run:")
    print("  1. v4.2 Conflict-first ObjectFile")
    print("  2. v5.1 Scaling sanity audit")
    print("=" * 70)

    scripts = [
        "scripts/run_svt_v4_2_conflict_first_object_file.py",
        "scripts/run_svt_v5_1_scaling_sanity_audit.py",
    ]

    for script in scripts:
        run_script(script)

    print("\n" + "=" * 70)
    print("Main diagnostic pipeline complete.")
    print("=" * 70)
    print("\nResults saved to:")
    print("  results/svt_v4_2_conflict_first_object_file/")
    print("  results/svt_v5_1_scaling_sanity_audit/")


if __name__ == "__main__":
    main()
