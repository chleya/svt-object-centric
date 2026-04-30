import sys
import os
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_command(cmd, description):
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"WARNING: Command failed with exit code {result.returncode}")
    return result.returncode == 0


def main():
    print("="*60)
    print("SVT-v2 Smoke Test Pipeline")
    print("="*60)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    success = True

    success &= run_command(
        "python data/generate_2d_motion.py --config configs/smoke.yaml",
        "Step 1: Generate 2D motion dataset"
    )

    success &= run_command(
        "python scripts/run_oracle_upper_bound.py --config configs/smoke.yaml",
        "Step 2: Run Oracle Upper Bound"
    )

    success &= run_command(
        "python scripts/run_knn_attack.py --config configs/smoke.yaml",
        "Step 3: Run k-NN Retrieval Attack v1"
    )

    print("\n" + "="*60)
    if success:
        print("All smoke tests passed!")
    else:
        print("Some steps failed. Check output above.")
    print("="*60)


if __name__ == "__main__":
    main()
