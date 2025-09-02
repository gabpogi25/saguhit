
import argparse, json, pandas as pd, os
from saguhit.shared_components import SharedGridData


ALGOS = {
    "bellman": "bellman_mdp_sungka",
    "sungka": "sungka",
    "greedy": "greedy_algorithm",
    "ga": "genetic_algorithm",
    "pso": "particle_swarm",
    "dcopf": "newton_raphson_opf"
}   

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--algos", required=True)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    shared = SharedGridData.from_edges(df)

    names = [a.strip() for a in args.algos.split(",") if a.strip()]
    os.makedirs(args.outdir, exist_ok=True)
    summary = {}

    for name in names:
        mod = __import__(ALGOS[name])
        res = mod.run(shared, df)
        out = {
            "algorithm": res.algorithm_name,
            "demand_met_kw": res.total_demand_satisfied_kw,
            "losses_kw": res.total_losses_kw,
            "efficiency_percent": res.system_efficiency_percent,
            "runtime_sec": res.computation_time_sec,
            "iterations": res.convergence_iterations
        }
        summary[name] = out
        with open(f"{args.outdir}/{name}.json","w") as f:
            json.dump(res.__dict__, f, default=lambda o: o.__dict__, indent=2)

    with open(f"{args.outdir}/summary.json","w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote comparison to {args.outdir}/summary.json")

if __name__ == "__main__":
    main()
