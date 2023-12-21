import pandas as pd
import numpy as np
import os

task = "LinkPrediction"
datasets = ["Cora"]
metrics = ["auc"]


def retrieve_results(df, metric=None, type_="max", split="Test"):
    if split == "Test" and type_ == "max":
        best_val_runs = dict(df.groupby("Run").max()[f"Val {metric}"])
        best_test_results = df[
            (df["Run"].isin(list(best_val_runs.keys())))
            & (df[f"Val {metric}"].isin(list(best_val_runs.values())))
        ][f"{split} {metric}"]
        return f"""{best_test_results.mean():.3f} ± {best_test_results.std():.4f}"""
    elif type_ == "max":
        return f"""{"{:0.4f}".format((df.groupby("Run")[f"{split} {metric}"].max().mean()))} ± {"{:0.4f}".format((df.groupby("Run")[f"{split} {metric}"].max().std()))}"""
    elif type_ == "last":
        return f"""{"{:0.4f}".format((df.groupby("Run")[f"{split} {metric}"].tail(1).mean()))} ± {"{:0.4f}".format((df.groupby("Run")[f"{split} {metric}"].tail(1).std()))}"""


def get_results(task, dataset, DIM):
    Node2Vec_results = pd.read_json(
        f"results/{task}/{dataset}/{DIM}/DownStream/results_Node2Vec_embedding.pth_False_False_False.json"
    )
    BASE_MLP = pd.read_json(f"results/{task}/{dataset}/{DIM}/DownStream/results_False_True_False_False.json")
    RANDOM = pd.read_json(f"results/{task}/{dataset}/{DIM}/DownStream/results_False_False_False_True.json")
    try:
        Spectral = pd.read_json(
            f"results/{task}/{dataset}/{DIM}/DownStream/results_False_False_True_False.json"
        )
    except:
        Spectral = None

    Shallow_Direct = pd.read_json(f"results/LinkPrediction/{dataset}/{DIM}/Shallow/results.json")

    shallow_results_for_runs = [
        x for x in os.listdir(f"results/{task}/{dataset}/{DIM}/DownStream/") if "shallow_embedding" in x
    ]
    for i, shallow_results in enumerate(shallow_results_for_runs):
        if i == 0:
            shallow_and_DownStream_results = pd.read_json(
                f"results/{task}/{dataset}/{DIM}/DownStream/{shallow_results}"
            )
            shallow_and_DownStream_results["Run"] = i + 1
        else:
            tmp = pd.read_json(f"results/{task}/{dataset}/{DIM}/DownStream/{shallow_results}")
            tmp["Run"] = i + 1
            shallow_and_DownStream_results = pd.concat([shallow_and_DownStream_results, tmp])

    # GNN Results
    GNN_PATH = f"results/{task}/{dataset}/{DIM}/GNN"
    GraphSage_results = GCN_results = None
    GraphSage_results_using_shallow = GCN_results_using_shallow = pd.DataFrame()
    GraphSage_results_using_spectral = GCN_results_using_spectral = pd.DataFrame()
    counter_GraphSage = counter_GCN = 1
    counter_GraphSage_spectral = 1
    counter_GCN_spectral = 1

    for file in os.listdir(GNN_PATH):
        if "GraphSage" in file and ((file.split("."))[0]).split("_")[-1] == "False":
            GraphSage_results = pd.read_json(GNN_PATH + "/" + file)
        elif "GCN" in file and ((file.split("."))[0]).split("_")[-1] == "False":
            GCN_results = pd.read_json(GNN_PATH + "/" + file)
        elif "GraphSage" in file and "shallow" in file:
            tmp_df = pd.read_json(GNN_PATH + "/" + file)
            tmp_df["Run"] = counter_GraphSage
            GraphSage_results_using_shallow = pd.concat([GraphSage_results_using_shallow, tmp_df])
            counter_GraphSage += 1
        elif "GCN" in file and "shallow" in file:
            tmp_df = pd.read_json(GNN_PATH + "/" + file)
            tmp_df["Run"] = counter_GCN
            GCN_results_using_shallow = pd.concat([GCN_results_using_shallow, tmp_df])
            counter_GCN += 1
        elif "GraphSage" in file and ((file.split("."))[0]).split("_")[-1] == "True":
            tmp_df = pd.read_json(GNN_PATH + "/" + file)
            tmp_df["Run"] = counter_GraphSage_spectral
            GraphSage_results_using_spectral = pd.concat([GraphSage_results_using_spectral, tmp_df])
            counter_GraphSage_spectral += 1
        elif "GCN" in file and ((file.split("."))[0]).split("_")[-1] == "True":
            tmp_df = pd.read_json(GNN_PATH + "/" + file)
            tmp_df["Run"] = counter_GCN_spectral
            GCN_results_using_spectral = pd.concat([GCN_results_using_spectral, tmp_df])
            counter_GCN_spectral += 1

    # combined results
    comb2_GCN = pd.read_json(f"results/LinkPrediction/{dataset}/{DIM}/combined/results_comb2_GCN.json")
    comb3_GCN = pd.read_json(f"results/LinkPrediction/{dataset}/{DIM}/combined/results_comb3_GCN.json")

    comb2_GraphSage = pd.read_json(
        f"results/LinkPrediction/{dataset}/{DIM}/combined/results_comb2_GraphSage.json"
    )
    comb3_GraphSage = pd.read_json(
        f"results/LinkPrediction/{dataset}/{DIM}/combined/results_comb3_GraphSage.json"
    )

    # Now create results table
    models = {
        "Random": RANDOM,
        "BASE_MLP": BASE_MLP,
        "Node2Vec": Node2Vec_results,
        "Shallow": Shallow_Direct,
        "shallow_and_DownStream_results": shallow_and_DownStream_results,
        "GraphSage": GraphSage_results,
        "GCN": GCN_results,
        "comb2_GCN": comb2_GCN,
        "comb3_GCN": comb3_GCN,
        "comb2_GraphSage": comb2_GraphSage,
        "comb3_GraphSage": comb3_GraphSage,
    }

    if GraphSage_results_using_shallow.shape[0] > 0:
        models["GraphSage_Using_Shallow"] = GraphSage_results_using_shallow

    if GCN_results_using_shallow.shape[0] > 0:
        models["GCN_Using_Shallow"] = GCN_results_using_shallow

    if GCN_results_using_spectral.shape[0] > 0:
        models["GCN_results_using_spectral"] = GCN_results_using_spectral

    if GraphSage_results_using_spectral.shape[0] > 0:
        models["GraphSage_results_using_spectral"] = GraphSage_results_using_spectral

    if Spectral.shape[0] > 0:
        models["Spectral"] = Spectral

    return models


def print_results(datasets, dim=16, type_="max", split="Test", task=None, metrics=None, to_latex=False):
    index_list = []
    for i, dataset in enumerate(datasets):
        models = get_results(task=task, dataset=dataset, DIM=dim)
        if i == 0:
            final_results = pd.DataFrame(columns=list(models.keys()))

        results_dict = {}
        for model in models:
            res = retrieve_results(models[model], metric=metrics[i], type_=type_, split=split)
            results_dict[model] = res
        final_results.loc[len(final_results), :] = results_dict
        index_list.append(dataset)

    final_results.index = index_list
    final_results = final_results.T
    if to_latex:
        print(metrics)
        print(final_results.to_latex())
    else:
        return final_results


print_results(
    datasets=datasets, dim=16, type_="max", split="Test", task=task, metrics=metrics, to_latex=False
)
