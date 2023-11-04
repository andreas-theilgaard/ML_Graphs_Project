from torch_geometric.datasets import Planetoid, Flickr, Amazon, Twitch
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected
from src.models.utils import set_seed
import torch_geometric.transforms as T
from torch_geometric.transforms import NormalizeFeatures


def get_link_data_split(data, num_test: float = 0.25, num_val: float = 0.25):
    set_seed(42)
    if data.is_directed():
        data.edge_index = to_undirected(data.edge_index)
    transform = RandomLinkSplit(
        is_undirected=True,
        num_test=num_test,
        num_val=num_val,
        add_negative_train_samples=False,
        split_labels=True,
    )
    train_data, val_data, test_data = transform(data)
    return train_data, val_data, test_data


class TorchGeometricDatasets:
    def __init__(self, dataset: str, task: str, model: str):
        self.dataset = dataset
        self.task = task
        self.model = model

    def get_dataset(self):
        if self.dataset in ["Cora", "CiteSeer", "PubMed"]:
            dataset = Planetoid(root="data", name=self.dataset, transform=self.use_transform())
        elif self.dataset in ["Flickr"]:
            dataset = Flickr(root="data", transform=self.use_transform())
        elif self.dataset in ["Computers", "Photo"]:
            dataset = Amazon(root="data", name=self.dataset, transform=self.use_transform())
        elif dataset in ["Twitch"]:
            dataset = Twitch(root="data", name="EN", transform=self.use_transform())
        return dataset

    def use_transform(self):
        if self.task == "NodeClassification" and self.model == "GNN":
            return T.ToSparseTensor()
        elif self.dataset in ["Cora"]:
            return NormalizeFeatures()
        return None
