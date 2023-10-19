import numpy as np

# loss | Train_acc | Val_acc | Test_acc
MLP_node_classification = np.load(
    "outputs/NodeClassification/ogbn-arxiv/DownStream/2023-10-09/22-12-48/results.npy"
)
GraphSage_node_classification = np.load(
    "outputs/NodeClassification/ogbn-arxiv/GNN/2023-10-10/18-26-36/results.npy"
)

import numpy as np

X = np.load(
    "outputs/LinkPrediction/ogbl-collab/DownStream/2023-10-11/08-25-38/results.npy"
)
round(X[-1, -1], 4)

if __name__ == "__main__":
    print("Node Classification ogbn-arxiv:")
    print(f"MLP on features: {MLP_node_classification[-1,-1]}")
    print(f"GraphSage on features: {GraphSage_node_classification[-1,-1]}")
