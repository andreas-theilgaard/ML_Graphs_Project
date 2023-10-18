import hydra
from omegaconf import OmegaConf
import logging
from src.data.get_data import DataLoader
from torch_geometric.utils import to_undirected
from src.models.utils import get_seeds

# Node Classification
from src.models.NodeClassification.mlp_nodeclass import mlp_node_classification
from src.models.NodeClassification.GNN import GNN_trainer

# Link Prediction
# from src.models.mlp_linkpredict import mlp_LinkPrediction
from src.models.LinkPrediction.mlp_linkpredict import mlp_LinkPrediction
from src.models.GNN.GNN_link import GNN_link_trainer

# Embeddings
from src.models.Node2Vec import Node2Vec
from src.models.shallow import ShallowTrainer


log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="../config")
def main(config):
    """
    This script is used for running all the experiments regarding this project.

    args:
        - config:
          yaml config file, please specify by using the --config-name flag
    """

    log.info(f"Starting experiment ... on {config.device}")
    log.info(
        f"\nConfigurations for current experiment:\n\nConfiguration: \n {OmegaConf.to_yaml(config)}"
    )
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    model_args = config.dataset[config.model_type]
    training_args = model_args.training
    save_path = hydra_cfg["runtime"]["output_dir"]

    # get seeds
    seeds = get_seeds(config.runs)

    # get data
    dataset = DataLoader(
        task_type=config.dataset.task,
        dataset=config.dataset.dataset_name,
        model_name=config.model_type,
    ).get_data()

    ###########################################
    # Downstream
    ###########################################
    if config.model_type == "DownStream":
        if config.dataset.task == "NodeClassification":
            mlp_node_classification(
                dataset=dataset,
                config=config,
                training_args=training_args,
                log=log,
                save_path=save_path,
                seeds=seeds,
            )
        elif config.dataset.task == "LinkPrediction":
            mlp_LinkPrediction(
                dataset=dataset,
                config=config,
                training_args=training_args,
                log=log,
                save_path=save_path,
                seeds=seeds,
            )

    ###########################################
    # Node2Vec
    ###########################################
    elif config.model_type == "Node2Vec":
        data = dataset[0]
        if data.is_directed():
            data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
        # If LinkPrediciton should problably do some more here
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

    ###########################################
    # GNN
    ###########################################
    elif config.model_type == "GNN":
        if config.dataset.task == "NodeClassification":
            GNN_trainer(
                dataset=dataset,
                config=config,
                training_args=training_args,
                log=log,
                save_path=save_path,
                seeds=seeds,
            )
        elif config.dataset.task == "LinkPrediction":
            GNN_link_trainer(
                dataset=dataset,
                config=config,
                training_args=training_args,
                save_path=save_path,
                log=log,
            )

    ###########################################
    # Shallow
    ###########################################
    elif config.model_type == "Shallow":
        Trainer = ShallowTrainer(
            config=config, training_args=training_args, save_path=save_path, log=log
        )
        Trainer.fit(dataset=dataset, seeds=seeds)

    ###########################################
    # Combined
    ###########################################
    elif config.model_type == "Combined":
        print("not implemented yet")

    else:
        raise ValueError(
            f"The specified model type, {config.model_type} is not yet supported."
        )

    # if config.debug:


if __name__ == "__main__":
    main()
