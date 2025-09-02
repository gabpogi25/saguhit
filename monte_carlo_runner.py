
import argparse, os, json, numpy as np, pandas as pd
from shared_components import SharedGridData

ALGOS = {
    "bellman": "bellman_mdp_sungka",
    "sungka": "sungka",
    "greedy": "greedy_algorithm",
    "ga": "genetic_algorithm",
    "pso": "particle_swarm",
    "dcopf": "newton_raphson_opf"
}

def perturb(df: pd.DataFrame, noise: float) -> pd.DataFrame:
    out = df.copy()
    if "Energy_kW" in out.columns:
        out["Energy_kW"] = out["Energy_kW"].fillna(0.0) * (1.0 + noise*(np.random.rand(len(out))*2 - 1))
        out["Energy_kW"] = out["Energy_kW"].clip(lower=0.0)
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--algos", required=True)
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--noise", type=float, default=0.1)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    base_df = pd.read_csv(args.input)
    os.makedirs(args.outdir, exist_ok=True)
    names = [a.strip() for a in args.algos.split(",") if a.strip()]

    agg = {name: [] for name in names}

    for t in range(args.trials):
        df = perturb(base_df, args.noise)
        shared = SharedGridData.from_edges(df)
        for name in names:
            mod = __import__(ALGOS[name])
            res = mod.run(shared, df)
            agg[name].append({
                "demand_met_kw": res.total_demand_satisfied_kw,
                "losses_kw": res.total_losses_kw,
                "efficiency_percent": res.system_efficiency_percent,
                "runtime_sec": res.computation_time_sec
            })

    summary = {}
    for name, rows in agg.items():
        import numpy as np
        def stats(key):
            arr = np.array([r[key] for r in rows])
            return {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max())
            }
        summary[name] = {k: stats(k) for k in ["demand_met_kw","losses_kw","efficiency_percent","runtime_sec"]}

    with open(f"{args.outdir}/mc_summary.json","w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote Monte Carlo summary to {args.outdir}/mc_summary.json")

if __name__ == "__main__":
    main()
