import subprocess
import ast


DISABLE_HYDRA = "hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled debug=True"
BASE = "python3 src/experiments/run_exps.py --config-name='base.yaml'"


endpoints = {
    "shallow": {
        "NodeClassification": {
            "laplacian": f"{BASE} dataset='ogbn-arxiv' {DISABLE_HYDRA} model_type='Shallow' runs=1 task='NodeClassification' dataset.Shallow.training.init='laplacian' dataset.Shallow.training.epochs=10",
            "random": f"{BASE} dataset='ogbn-arxiv' {DISABLE_HYDRA} model_type='Shallow' runs=1 task='NodeClassification' dataset.Shallow.training.init='random' dataset.Shallow.training.epochs=10",
        },
        "LinkPrediction": {
            "laplacian": f"{BASE} dataset='ogbl-collab' {DISABLE_HYDRA} model_type='Shallow' runs=1 task='LinkPrediction' dataset.Shallow.training.init='laplacian' dataset.Shallow.training.epochs=10",
            "random": f"{BASE} dataset='ogbl-collab' {DISABLE_HYDRA} model_type='Shallow' runs=1 task='LinkPrediction' dataset.Shallow.training.init='random' dataset.Shallow.training.epochs=10",
        },
    },
    "DownStream": {
        "NodeClassification": {
            "BaselineMLP": f"{BASE} dataset='ogbn-arxiv' {DISABLE_HYDRA} model_type=DownStream runs=1 task='NodeClassification' dataset.DownStream.training.epochs=10"
        },
        "LinkPrediction": {
            "BaselineMLP": f"{BASE} dataset='ogbl-collab' {DISABLE_HYDRA} model_type=DownStream runs=1 task='LinkPrediction' dataset.DownStream.training.epochs=10"
        },
    },
    "GNN": {
        "NodeClassification": {
            "GraphSage": f"{BASE} dataset='ogbn-arxiv' {DISABLE_HYDRA} model_type=GNN runs=1 task='NodeClassification' dataset.GNN.training.epochs=10 dataset.GNN.model='GraphSage'",
            "GCN": f"{BASE} dataset='ogbn-arxiv' {DISABLE_HYDRA} model_type=GNN runs=1 task='NodeClassification' dataset.GNN.training.epochs=10 dataset.GNN.model='GCN'",
        },
        "LinkPrediction": {
            "GraphSage": f"{BASE} dataset='ogbl-collab' {DISABLE_HYDRA} model_type=GNN runs=1 task='LinkPrediction' dataset.GNN.training.epochs=2 dataset.GNN.model='GraphSage'",
            "GCN": f"{BASE} dataset='ogbl-collab' {DISABLE_HYDRA} model_type=GNN runs=1 task='LinkPrediction' dataset.GNN.training.epochs=2 dataset.GNN.model='GCN'",
        },
    },
}


######################################
# Node Classification tests
######################################


def test_shallow_classification():
    # Laplacian
    completed_process = subprocess.run(
        endpoints["shallow"]["NodeClassification"]["laplacian"],
        capture_output=True,
        shell=True,
        text=True,
        check=True,
    )
    output = completed_process.stdout
    output = ast.literal_eval(output)
    assert output["loss"] == 0.5297333598136902
    assert output["acc"] == 0.6928158402442932
    # Random
    completed_process = subprocess.run(
        endpoints["shallow"]["NodeClassification"]["random"],
        capture_output=True,
        shell=True,
        text=True,
        check=True,
    )
    output = completed_process.stdout
    output = ast.literal_eval(output)
    assert output["loss"] == 0.5949175953865051
    assert output["acc"] == 0.59163898229599


def test_downstream_classification():
    completed_process = subprocess.run(
        endpoints["DownStream"]["NodeClassification"]["BaselineMLP"],
        capture_output=True,
        shell=True,
        text=True,
        check=True,
    )
    output = completed_process.stdout
    output = ast.literal_eval(output)
    assert output["loss"] == 1.993174433708191
    assert output["Test acc"] == 0.43513774869864


def test_gnn_classification():
    completed_process = subprocess.run(
        endpoints["GNN"]["NodeClassification"]["GraphSage"],
        capture_output=True,
        shell=True,
        text=True,
        check=True,
    )
    output = completed_process.stdout
    output = ast.literal_eval(output)
    assert output["loss"] == 1.4672894477844238
    assert output["Test acc"] == 0.5953953459662984
    completed_process = subprocess.run(
        endpoints["GNN"]["NodeClassification"]["GCN"],
        capture_output=True,
        shell=True,
        text=True,
        check=True,
    )
    output = completed_process.stdout
    output = ast.literal_eval(output)
    assert output["loss"] == 1.3775808811187744
    assert output["Test acc"] == 0.5042692837890665


######################################
# LinkPredictions tests
######################################


def test_downstream_linkprediction():
    completed_process = subprocess.run(
        endpoints["DownStream"]["LinkPrediction"]["BaselineMLP"],
        capture_output=True,
        shell=True,
        text=True,
        check=True,
    )
    output = completed_process.stdout
    output = ast.literal_eval(output)
    assert output["loss"] == 0.504451534952073
    assert output["Test hits@50"] == 0.02900990740141164


def test_shallow_linkprediction():
    # Laplacian
    completed_process = subprocess.run(
        endpoints["shallow"]["LinkPrediction"]["laplacian"],
        capture_output=True,
        shell=True,
        text=True,
        check=True,
    )
    output = completed_process.stdout
    output = ast.literal_eval(output)
    assert output["loss"] == 0.5598081350326538
    assert output["acc"] == 0.7010598182678223
    # Random
    completed_process = subprocess.run(
        endpoints["shallow"]["LinkPrediction"]["random"],
        capture_output=True,
        shell=True,
        text=True,
        check=True,
    )
    output = completed_process.stdout
    output = ast.literal_eval(output)
    assert output["loss"] == 0.6056017875671387
    assert output["acc"] == 0.5937978029251099


# def test_gnn_linkprediction():
#     completed_process = subprocess.run(endpoints['GNN']['LinkPrediction']['GraphSage'], capture_output=True, shell=True,text=True, check=True)
#     output = completed_process.stdout
#     output = ast.literal_eval(output) #1.0114138905597612
#     assert output['loss'] == 1.0102661845384884
#     assert output['hits@50'] == 0.006259578233935548
#     completed_process = subprocess.run(endpoints['GNN']['LinkPrediction']['GCN'], capture_output=True, shell=True,text=True, check=True)
#     output = completed_process.stdout
#     output = ast.literal_eval(output)
#     assert output['loss'] == 1.0085030351438709
#     assert output['hits@50'] == 0.033067840877204345
#
