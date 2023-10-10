import hydra
from omegaconf import OmegaConf
import logging
import torch
from src.data.get_data import DataLoader
from src.models.mlp_nodeclass import mlp_node_classification
from src.models.EmbeddingNetworks.Node2Vec import Node2Vec
from src.models.GNN.GNN import GNN_trainer
from torch_geometric.utils import to_undirected
import random

log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="../config")
def main(config):
    log.info(f"Starting run ... on {config.device}")
    print(
        f"\nConfigurations for current run:\n\nConfiguration: \n {OmegaConf.to_yaml(config)}"
    )
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    training_args = config.model.training
    save_path = hydra_cfg["runtime"]["output_dir"]
    # seed initailization
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset = DataLoader(
        task_type=config.task,
        dataset=config.dataset,
        model_name=config.model.model_name,
    ).get_data()

    if config.model.model_name == "DownStream":
        if config.task == "NodeClassification":
            mlp_node_classification(
                dataset=dataset,
                config=config,
                training_args=training_args,
                log=log,
                save_path=save_path,
            )
        elif config.task == "LinkPrediction":
            print("Not implemented yet")

    elif config.model.model_name == "Node2Vec":
        data = dataset[0]
        if data.is_directed():
            data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
        model = Node2Vec(
            edge_index=data.edge_index,
            device=config.device,
            save_path=save_path,
            embedding_dim=training_args.embedding_dim,
            walk_length=training_args.walk_length,
            walks_per_node=training_args.walks_per_node,
            context_size=training_args.context_size,
            num_negative_samples=training_args.num_negative_samples,
            sparse=training_args.sparse,
        )
        model.train(
            batch_size=training_args.batch_size,
            epochs=training_args.epochs,
            lr=training_args.lr,
            num_workers=training_args.num_workers,
        )
    elif config.model.model_name == "GNN":
        GNN_trainer(dataset, config, training_args, save_path, log)

    elif config.model.model_name == "Shallow":
        print("not implemented yet")

    elif config.model.model_name == "Combined":
        print("not implemented yet")


if __name__ == "__main__":
    main()
