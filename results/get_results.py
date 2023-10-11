import numpy as np

# loss | Train_acc | Val_acc | Test_acc
MLP_node_classification = np.load(
    "outputs/NodeClassification/ogbn-arxiv/DownStream/2023-10-09/22-12-48/results.npy"
)
GraphSage_node_classification = np.load(
    "outputs/NodeClassification/ogbn-arxiv/GNN/2023-10-10/18-26-36/results.npy"
)


if __name__ == "__main__":
    print("Node Classification ogbn-arxiv:")
    print(f"MLP on features: {MLP_node_classification[-1,-1]}")
    print(f"GraphSage on features: {GraphSage_node_classification[-1,-1]}")
