
import argparse
import sys
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Run SaGUHiT Optimization Modes")
    parser.add_argument("--mode", choices=["comparison", "montecarlo", "single"], required=True,
                        help="Run mode: comparison (all algos), montecarlo (robustness), single (one algo)")
    parser.add_argument("--algo", type=str, default=None,
                        help="Algorithm name if running single mode (e.g. bellman, greedy, pso, genetic, opf)")
    parser.add_argument("--trials", type=int, default=20,
                        help="Number of trials for Monte Carlo mode")
    parser.add_argument("--noise", type=float, default=0.05,
                        help="Noise level for Monte Carlo mode (e.g. 0.05 = ±5%)")

    args = parser.parse_args()

    if args.mode == "comparison":
        print("🔹 Running comparison mode...")
        subprocess.run([sys.executable, "comparison_runner.py"])

    elif args.mode == "montecarlo":
        print(f"🔹 Running Monte Carlo mode with {args.trials} trials, noise={args.noise}...")
        subprocess.run([sys.executable, "monte_carlo_runner.py", 
                        "--trials", str(args.trials), "--noise", str(args.noise)])

    elif args.mode == "single":
        if not args.algo:
            print(" You must specify --algo when using single mode")
            sys.exit(1)
        print(f"🔹 Running single algorithm: {args.algo}")
        subprocess.run([sys.executable, "runner.py", "--algo", args.algo])

if __name__ == "__main__":
    main()
