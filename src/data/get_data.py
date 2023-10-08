from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
import torch


class DataLoader:
    def __init__(
        self,
        model_name: str,
        task_type: str = "NodeClassification",
        dataset: str = "ogbn-arxiv",
    ):
        self.task_type = task_type
        self.dataset = dataset
        self.model_name = model_name
        self.assert_arguments()

    def assert_arguments(self):
        assert self.task_type in [
            "NodeClassification",
            "LinkPrediction",
        ], f"Expect task_type to be either 'NodeClassification' or 'LinkPrediction' but received {task_type}"
        if self.task_type == "NodeClassification":
            assert self.dataset in [
                "ogbn-arxiv",
                "ogbn-products",
            ], f"Expect NodeClassification dataset to be one of ['ogbn-arxiv'] but received {self.dataset}"
        if self.task_type == "LinkPrediction":
            assert self.dataset in [
                "ogbn-arxiv"
            ], f"Expect LinkPrediction dataset to be one of ['ogbn-arxiv'] but received {self.dataset}"

    def get_NodeClassification_dataset(self):
        dataset = (
            PygNodePropPredDataset(name=self.dataset, root="data")
            if self.model_name != "GNN"
            else PygNodePropPredDataset(
                name=self.dataset, root="data", transform=T.ToSparseTensor()
            )
        )
        self.dataset_summary(dataset)
        # data = dataset[0]
        # if not data.is_undirected():
        #    data.edge_index = to_undirected(data.edge_index,num_nodes=data.num_nodes)
        return dataset

    def dataset_summary(self, dataset):
        summary = f"""\n    ===========================
    Dataset: {dataset.name}:
    ===========================
    Number of graphs: {len(dataset)} \n
    Number of features: {dataset.num_features} \n
    Number of classes: {dataset.num_classes}
        """
        if len(dataset) == 1:
            data = dataset[0]
            summary += f"""
    Number of nodes: {data.num_nodes} \n
    Number of edges: {data.num_edges} \n
    Is undirected: {data.is_undirected()}
            """
        print(summary)

    def get_data(self):
        if self.task_type == "NodeClassification":
            return self.get_NodeClassification_dataset()
        elif self.task_type == "LinkPrediction":
            print("Not implemented yet")
