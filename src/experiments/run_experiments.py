import hydra
from omegaconf import OmegaConf
import logging
import torch
from src.data.get_data import DataLoader
from src.models.EmbeddingNetworks.Node2Vec import Node2Vec

# from src.models.ShallowNetworks.Shallow import ShallowEmbeddings
from src.models.GNN.GNN import GNN
import argparse
from torch_geometric.utils import to_undirected
from src.models.mlp import MLP_model, MLP_LinkPrediction
from ogb.nodeproppred import Evaluator as eval_classifier
from ogb.linkproppred import Evaluator as eval_linkPredictor
import random
from tqdm import tqdm

log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="../config")
def main(config):
    log.info("Starting run ...")
    print(
        f"\nConfigurations for current run:\n\nConfiguration: \n {OmegaConf.to_yaml(config)}"
    )
    training_args = config.model.training

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
    data = dataset[0]

    if config.model.model_name == "Node2Vec":
        if data.is_directed():
            data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
        save_path = f"models/embeddings/{config.dataset}/{config.model.model_name}/"
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

    elif config.model.model_name == "Shallow":
        print("Run shallow")

    elif config.model.model_name == "GNN":
        evaluator = Evaluator(name=config.dataset)
        data.adj_t = data.adj_t.to_symmetric()
        split_idx = dataset.get_idx_split()
        GNN_object = GNN(
            GNN_type=config.model.model,
            task=config.task,
            in_channels=data.num_features,
            hidden_channels=training_args.hidden_channels,
            out_channels=dataset.num_classes,
            dropout=training_args.dropout,
            num_layers=training_args.num_layers,
        )
        model = GNN_object.get_gnn_model()
        model.reset_parameters()
        model.to(config.device)
        train_idx = split_idx["train"].to(config.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=training_args.lr)
        prog_bar = tqdm(range(training_args.epochs))

        for epoch in prog_bar:
            loss = GNN_object.train(model, data, train_idx, optimizer)
            result = GNN_object.test(model, data, split_idx, evaluator)
            train_acc, valid_acc, test_acc = result
            prog_bar.set_postfix(
                {
                    "Train Loss": loss,
                    "Train Acc.": train_acc,
                    "Val Acc.": valid_acc,
                    "Test Acc.": test_acc,
                }
            )
            if epoch % 10 == 0:
                log.info(
                    f"Train: {100*train_acc}, Valid: {100*valid_acc}, Test: {100*test_acc}"
                )
        model.train()

    elif config.model.model_name == "DownStream":
        if config.task == "LinkPrediction":
            split_edge = dataset.get_edge_split()
            if config.model.saved_embeddings and config.model.using_features:
                embedding = torch.load(
                    config.model.saved_embeddings, map_location=config.device
                )
                x = torch.cat([data.x, embedding], dim=-1)
            if config.model.saved_embeddings and not config.model.using_features:
                embedding = torch.load(
                    config.model.saved_embeddings, map_location=config.device
                )
                x = embedding
            if not config.model.saved_embeddings and config.model.using_features:
                x = data.x
            X = x.to(config.device)

            evaluator = eval_linkPredictor(name=config.dataset)
            model = MLP_LinkPrediction(
                task=config.task,
                device=config.device,
                in_channels=X.shape[-1],
                hidden_channels=training_args.hidden_channels,
                out_channels=1,
                num_layers=training_args.num_layers,
                dropout=training_args.dropout,
                log=log,
            )
            model.fit(
                X=X,
                split_edge=split_edge,
                lr=training_args.lr,
                batch_size=training_args.batch_size,
                epochs=training_args.epochs,
                evaluator=evaluator,
            )

        if config.task == "NodeClassification":
            split_idx = dataset.get_idx_split()

            if config.model.saved_embeddings and config.model.using_features:
                embedding = torch.load(
                    config.model.saved_embeddings, map_location=config.device
                )
                x = torch.cat([data.x, embedding], dim=-1)
            if config.model.saved_embeddings and not config.model.using_features:
                embedding = torch.load(
                    config.model.saved_embeddings, map_location=config.device
                )
                x = embedding
            if not config.model.saved_embeddings and config.model.using_features:
                x = data.x
            X = x.to(config.device)
            y = data.y.to(config.device)

            evaluator = Evaluator(name=config.dataset)

            model = MLP_model(
                device=config.device,
                in_channels=x.shape[-1],
                hidden_channels=training_args.hidden_channels,
                out_channels=dataset.num_classes,
                num_layers=training_args.num_layers,
                dropout=training_args.dropout,
                log=log,
            )
            model.fit(
                X=X,
                y=y,
                epochs=training_args.epochs,
                split_idx=split_idx,
                evaluator=evaluator,
                lr=training_args.lr,
            )

    elif config.model.model_name == "Combined":
        print("Run combined")


if __name__ == "__main__":
    main()
