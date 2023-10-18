import subprocess

if __name__ == "__main__":
    # Default MLP ogbn-arxiv
    # print("Default MLP ogbn-arxiv")
    # subprocess.call("python3 src/experiments/run_exps.py --config-name='base.yaml' dataset='ogbn-arxiv' model_type='DownStream'",shell=True)

    # Default GraphSage ogbn-arxiv
    # print("Default GraphSage ogbn-arxiv")
    # subprocess.call("python3 src/experiments/run_exps.py --config-name='base.yaml' dataset='ogbn-arxiv' model_type='GNN' dataset.GNN.model='GraphSage'",shell=True)

    # Default GraphSage ogbn-arxiv
    # print("Default GCN ogbn-arxiv")
    # subprocess.call("python3 src/experiments/run_exps.py --config-name='base.yaml' dataset='ogbn-arxiv' model_type='GNN' dataset.GNN.model='GCN'",shell=True)

    # Default Node2Vex ogbn-arxiv
    # print("Default Node2Vec ogbn-arxiv")
    # subprocess.call("python3 src/experiments/run_exps.py --config-name='base.yaml' dataset='ogbn-arxiv' model_type='Node2Vec'",shell=True)

    # Default Shallow ogbn-arxiv
    print("Default Shallow ogbn-arxiv")
    subprocess.call(
        "python3 src/experiments/run_exps.py --config-name='base.yaml' dataset='ogbn-arxiv' model_type='Shallow' dataset.traning.init='laplacian'",
        shell=True,
    )

    # print("Default MLP ogbn-arxiv")
    # subprocess.call("python3 src/experiments/run_exps.py --config-name='base.yaml' dataset='ogbl-collab' model_type='DownStream'",shell=True)
