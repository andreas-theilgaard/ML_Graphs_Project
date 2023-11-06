import argparse
import subprocess
from omegaconf import OmegaConf
import os

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

    args = parser.parse_args()

    CONFIG_SETUP = f"{BASE} dataset={args.dataset} device={args.device} +save_to_folder='results'"
    conf = OmegaConf.load(f"src/config/dataset/{args.dataset}.yaml")

    # Define paths
    Node2Vec_path = f"results/{conf.task}/{conf.dataset_name}/Node2Vec/Node2Vec_embedding.pth"
    Shallow_path = f"results/{conf.task}/{conf.dataset_name}/Shallow"

    # Run Baseline MLP Using Features
    subprocess.call(
        f"{CONFIG_SETUP} model_type='DownStream' runs={args.runs} dataset.DownStream.saved_embeddings=False dataset.DownStream.using_features=True",
        shell=True,
    )

    # Run Node2Vec
    subprocess.call(f"{CONFIG_SETUP} model_type='Node2Vec' runs=1", shell=True)
    subprocess.call(
        f"{CONFIG_SETUP} model_type='DownStream' runs={args.runs} dataset.DownStream.saved_embeddings={Node2Vec_path} dataset.DownStream.using_features=False",
        shell=True,
    )

    # Run Shallow
    subprocess.call(f"{CONFIG_SETUP} model_type='Shallow' runs={args.runs}", shell=True)
    shallow_embeddings = (
        [x for x in os.listdir(Shallow_path) if (".pth" in x and "best" in x)]
        if conf.task == "LinkPrediction"
        else [x for x in os.listdir(Shallow_path) if (".pth" in x)]
    )
    for shallow_embedding in shallow_embeddings:
        subprocess.call(
            f"{CONFIG_SETUP} model_type='DownStream' runs=1 dataset.DownStream.saved_embeddings={Shallow_path+'/'+shallow_embedding} dataset.DownStream.using_features=False",
            shell=True,
        )

    # Run GNN
    subprocess.call(
        f"{CONFIG_SETUP} model_type='GNN' runs={args.runs} dataset.GNN.model='GraphSage'", shell=True
    )
    subprocess.call(f"{CONFIG_SETUP} model_type='GNN' runs={args.runs} dataset.GNN.model='GCN'", shell=True)
    for shallow_embedding in shallow_embeddings:
        subprocess.call(
            f"{CONFIG_SETUP} model_type='GNN' runs=1 dataset.GNN.extra_info={Shallow_path+'/'+shallow_embedding} dataset.GNN.model='GraphSage'",
            shell=True,
        )
        subprocess.call(
            f"{CONFIG_SETUP} model_type='GNN' runs=1 dataset.GNN.extra_info={Shallow_path+'/'+shallow_embedding} dataset.GNN.model='GCN'",
            shell=True,
        )

    # Run spectral method
    subprocess.call(
        f"{CONFIG_SETUP} model_type='DownStream' runs={args.runs} dataset.DownStream.saved_embeddings=False dataset.DownStream.using_features=False use_spectral=True",
        shell=True,
    )

    # Run combined method
