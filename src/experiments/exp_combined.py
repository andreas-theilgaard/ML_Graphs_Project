import argparse
import subprocess

BASE = "~/miniconda3/envs/act_DLG/bin/python src/experiments/run_exps.py --config-name='base.yaml'"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script is used reproducing the results for a certain dataset"
    )
    # dataset
    parser.add_argument(
        "--dataset", default="ogbn-arxiv", type=str, required=False, help="Choose which dataset to use"
    )
    # device
    parser.add_argument("--device", default="cpu", type=str, required=False, choices=["cpu", "cuda", "mps"])
    # runs
    parser.add_argument(
        "--runs",
        default=10,
        type=int,
        required=False,
        help="Number of times the experiments should be repeated",
    )

    parser.add_argument("--task", default="LinkPrediction", type=str, required=False)

    args = parser.parse_args()

    CONFIG_SETUP = f"{BASE} dataset={args.dataset} device={args.device}"

    # Run combined method
    subprocess.call(
        f"{CONFIG_SETUP} model_type='combined' runs={args.runs} dataset.combined.type='type1'", shell=True
    )
    subprocess.call(
        f"{CONFIG_SETUP} model_type='combined' runs={args.runs} dataset.combined.type='type2'", shell=True
    )

    if args.task == "LinkPrediction":
        subprocess.call(
            f"{CONFIG_SETUP} model_type='combined' runs={args.runs} dataset.combined.type='type3'", shell=True
        )
        subprocess.call(
            f"{CONFIG_SETUP} model_type='combined' runs={args.runs} dataset.combined.type='type4'", shell=True
        )
