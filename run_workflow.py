import subprocess
import argparse

def run_command(cmd: str):
    print(f"\n> Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--affordance", default="grasp",
                        help="Which affordance to evaluate (default: grasp)")
    parser.add_argument("--n_objects", type=int, default=20,
                        help="Number of objects for PR curves and ablation")
    parser.add_argument("--figures", default="all",
                        choices=["comparison", "ablation", "pr", "all"],
                        help="Which figures to generate")
    args = parser.parse_args()

    # Step 1: Run LASO
    run_command(f"python experiments/run_laso.py --affordance {args.affordance}")

    # Step 2: Run ablation
    run_command(f"python experiments/ablation.py --affordances {args.affordance} --n_objects {args.n_objects}")

    # Step 3: Generate figures
    run_command(f"python experiments/make_figures.py --figure {args.figures} --affordance {args.affordance} --n-objects {args.n_objects}")
