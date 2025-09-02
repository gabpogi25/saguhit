
import argparse, pandas as pd, json, time
from shared_components import SharedGridData

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
    p.add_argument("--algorithm", required=True, choices=list(ALGOS.keys()))
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    shared = SharedGridData.from_edges(df)

    mod = __import__(ALGOS[args.algorithm])
    res = mod.run(shared, df)

    import os
    os.makedirs(args.outdir, exist_ok=True)
    with open(f"{args.outdir}/{args.algorithm}.json","w") as f:
        json.dump(res.__dict__, f, default=lambda o: o.__dict__, indent=2)
    print(f"Saved: {args.outdir}/{args.algorithm}.json")

if __name__ == "__main__":
    main()
