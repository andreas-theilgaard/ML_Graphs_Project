from src.models.models import MLP_model
from ogb.nodeproppred import Evaluator
import torch
from src.models.utils import create_path, cur_time
import numpy as np


def mlp_node_classification(dataset, config, training_args, log, save_path):
    data = dataset[0]
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
    save_results = model.fit(
        X=X,
        y=y,
        epochs=training_args.epochs,
        split_idx=split_idx,
        evaluator=evaluator,
        lr=training_args.lr,
    )

    np.save(save_path + "/results.npy", save_results)
