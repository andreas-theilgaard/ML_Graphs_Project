from src.models.NodeClassification.mlp import MLP_model
from ogb.nodeproppred import Evaluator
import torch
import numpy as np
from src.models.utils import set_seed
from src.models.utils import prepare_metric_cols


def mlp_node_classification(dataset, config, training_args, log, save_path, seeds, Logger):
    """
    Function that instanties and runs a MLP model for the node classification task

    args:
        dataset:
            torch geometric dataset
        config:
            config form yaml file
        training_args:
            traning arguments from config, used for shorten the reference to these args
        save_path:
            path to the current hydra folder
        seeds:
            list of seeds that will be used for the current experiment
        Logger:
            the Logger class as defined in src/models/logger.py

    """
    data = dataset[0]
    split_idx = dataset.get_idx_split()

    if (
        config.dataset[config.model_type].saved_embeddings
        and config.dataset[config.model_type].using_features
    ):
        embedding = torch.load(config.model.saved_embeddings, map_location=config.device)
        x = torch.cat([data.x, embedding], dim=-1)
    if (
        config.dataset[config.model_type].saved_embeddings
        and not config.dataset[config.model_type].using_features
    ):
        embedding = torch.load(config.model.saved_embeddings, map_location=config.device)
        x = embedding
    if (
        not config.dataset[config.model_type].saved_embeddings
        and config.dataset[config.model_type].using_features
    ):
        x = data.x
    X = x.to(config.device)
    y = data.y.to(config.device)

    evaluator = Evaluator(name=config.dataset.dataset_name)

    for seed in seeds:
        set_seed(seed=seed)
        Logger.start_run()

        model = MLP_model(
            device=config.device,
            in_channels=x.shape[-1],
            hidden_channels=training_args.hidden_channels,
            out_channels=dataset.num_classes,
            num_layers=training_args.num_layers,
            dropout=training_args.dropout,
            log=log,
            logger=Logger,
        )

        model.fit(
            X=X,
            y=y,
            epochs=training_args.epochs,
            split_idx=split_idx,
            evaluator=evaluator,
            lr=training_args.lr,
        )
        Logger.end_run()

    Logger.save_results(save_path + "/results.json")
    Logger.get_statistics(
        metrics=prepare_metric_cols(config.dataset.metrics),
        directions=["-", "+", "+", "+"],
    )
