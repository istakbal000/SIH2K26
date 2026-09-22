import subprocess
import sys
from pathlib import Path
import argparse

BASE_DIR = Path(__file__).resolve().parent.parent

def run_script(script_name: str, args: list = None):
    print(f"\n{'='*60}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*60}")
    script_path = BASE_DIR / "src" / script_name
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    result = subprocess.run(cmd, cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"\nFAILED: {script_name} exited with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\nDONE: {script_name}")
    return result

def main():
    parser = argparse.ArgumentParser(description="Run complete fire detection ML pipeline")
    parser.add_argument("--raw", type=str, default=None, help="Path to raw CSV file (default: data/raw/fire_data.csv)")
    parser.add_argument("--time-based", action="store_true", help="Use chronological train/val/test split instead of random shuffle")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing step")
    parser.add_argument("--skip-split", action="store_true", help="Skip train/test split step")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training step")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation step")
    args = parser.parse_args()

    if not args.skip_preprocess:
        raw_args = [f"--raw {args.raw}"] if args.raw else []
        run_script("preprocess.py", raw_args)

    if not args.skip_split:
        split_args = []
        if args.time_based:
            split_args.extend(["--mode", "time"])
        run_script("split_data.py", split_args)

    if not args.skip_train:
        run_script("train.py")

    if not args.skip_eval:
        run_script("evaluate.py")

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()