import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models.utils import set_seed, get_negative_samples
from src.models.combined.combined_link.combined_link_utils import (
    LinkPredictor,
    SAGE,
    test_indi_with_predictor,
    test_joint_with_predictor,
    get_split_edge,
    ModelWeights,
    GCN,
    predict,
)
from src.models.shallow import ShallowModel, initialize_embeddings
from src.models.metrics import METRICS
from src.models.utils import prepare_metric_cols
import numpy as np
from src.models.utils import create_path


def warm_train(
    shallow,
    deep,
    predictor,
    data_deep,
    optimizer_shallow,
    optimizer_deep,
    data_shallow,
    criterion,
    batch_size=65538,
    config=None,
):
    shallow.train()
    deep.train()
    predictor.train()

    pos_edge_index = data_shallow
    pos_edge_index = pos_edge_index.to(data_deep.x.device)

    for perm in DataLoader(range(pos_edge_index.size(1)), batch_size, shuffle=True):
        optimizer_shallow.zero_grad()
        optimizer_deep.zero_grad()

        # W = deep(data_deep.x, data_deep.adj_t)
        W = deep(
            data_deep.x, data_deep.adj_t
        )  # if config.dataset.dataset_name == 'ogbl-collab' else deep(data_deep.x, data_deep.edge_index)
        edge = pos_edge_index[:, perm]

        # shallow
        pos_out_shallow = shallow(edge[0], edge[1])

        # deep
        pos_out_deep = predictor(W[edge[0]], W[edge[1]])

        # Negative edges

        neg_edge_index = get_negative_samples(edge, data_deep.x.shape[0], edge.size(1))
        neg_edge_index = neg_edge_index.to(data_deep.x.device)
        neg_out_shallow = shallow(neg_edge_index[0], neg_edge_index[1])
        neg_out_deep = predictor(W[neg_edge_index[0]], W[neg_edge_index[1]])

        # concat positive and negative predictions
        total_predictions_shallow = torch.cat([pos_out_shallow, neg_out_shallow], dim=0)
        total_predictions_deep = torch.cat([pos_out_deep, neg_out_deep], dim=0)
        y_shallow = torch.cat(
            [torch.ones(pos_out_shallow.size(0)), torch.zeros(neg_out_shallow.size(0))], dim=0
        ).to(data_deep.x.device)
        y_deep = (
            torch.cat([torch.ones(pos_out_deep.size(0)), torch.zeros(neg_out_deep.size(0))], dim=0)
            .to(data_deep.x.device)
            .unsqueeze(1)
        )

        # calculate loss
        loss_shallow = criterion(total_predictions_shallow, y_shallow)
        loss_deep = criterion(total_predictions_deep, y_deep)

        # optimization step
        loss_shallow.backward()
        optimizer_shallow.step()

        # optimization step
        loss_deep.backward()

        torch.nn.utils.clip_grad_norm_(deep.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer_deep.step()

    return (loss_shallow, loss_deep)


def fit_warm_start(
    warm_start,
    shallow,
    deep,
    data_deep,
    data_shallow,
    predictor,
    optimizer_shallow,
    optimizer_deep,
    criterion,
    split_edge,
    evaluator,
    batch_size,
    config,
    log,
    training_args,
):
    prog_bar = tqdm(range(warm_start))
    for i in prog_bar:
        loss_shallow, loss_deep = warm_train(
            shallow=shallow,
            deep=deep,
            predictor=predictor,
            data_deep=data_deep,
            data_shallow=data_shallow,
            optimizer_shallow=optimizer_shallow,
            optimizer_deep=optimizer_deep,
            criterion=criterion,
            config=config,
            batch_size=training_args.batch_size,
        )

        prog_bar.set_postfix({"Shallow L": loss_shallow.item(), "Deep L": loss_deep.item()})

        if i % 10 == 0:
            results_shallow, results_deep = test_indi_with_predictor(
                shallow=shallow,
                deep=deep,
                linkpredictor=predictor,
                split_edge=split_edge,
                x=data_deep.x,
                adj_t=data_deep.adj_t,
                evaluator=evaluator,
                batch_size=batch_size,
            )

            log.info(
                f"Shallow: Train {config.dataset.track_metric}:{results_shallow['train'][config.dataset.track_metric]}, Val {config.dataset.track_metric}:{results_shallow['val'][config.dataset.track_metric]}, Test {config.dataset.track_metric}:{results_shallow['test'][config.dataset.track_metric]}"
            )
            log.info(
                f"Deep: Train {config.dataset.track_metric}:{results_deep['train'][config.dataset.track_metric]}, Val {config.dataset.track_metric}:{results_deep['val'][config.dataset.track_metric]}, Test {config.dataset.track_metric}:{results_deep['test'][config.dataset.track_metric]}"
            )


def fit_combined2_link(config, dataset, training_args, Logger, log, seeds, save_path):
    data = dataset[0]
    undirected = True
    if data.is_directed():
        undirected = False
        data.edge_index = to_undirected(data.edge_index)

    data = data.to(config.device)
    data_shallow, split_edge, data_splits = get_split_edge(
        data=data, dataset=dataset, config=config, training_args=training_args
    )

    if "edge_weight" in data:  #
        data.edge_weight = data.edge_weight.view(-1).to(torch.float)  #
    data_deep = T.ToSparseTensor()(data)
    if not undirected:  #
        data_deep.adj_t = data_deep.adj_t.to_symmetric()  #
    data_deep = data_deep.to(config.device)

    for counter, seed in enumerate(seeds):
        set_seed(seed)
        Logger.start_run()

        ##### Setup models #####
        init_embeddings = initialize_embeddings(
            num_nodes=data.x.shape[0],
            data=data_shallow,
            dataset=dataset,
            method=training_args.init,
            dim=training_args.embedding_dim,
            for_link=True,
            edge_split=split_edge,
        )
        init_embeddings = init_embeddings.to(config.device)
        shallow = ShallowModel(
            num_nodes=data.x.shape[0],
            embedding_dim=training_args.embedding_dim,
            beta=training_args.init_beta,
            init_embeddings=init_embeddings,
            decoder_type=training_args.decode_type,
            device=config.device,
        ).to(config.device)
        if training_args.deep_model == "GraphSage":
            deep = SAGE(
                in_channels=data.num_features,
                hidden_channels=training_args.deep_hidden_channels,
                out_channels=training_args.deep_out_dim,
                num_layers=training_args.deep_num_layers,
                dropout=training_args.deep_dropout,
            ).to(config.device)
        elif training_args.deep_model == "GCN":
            deep = GCN(
                in_channels=data.num_features,
                hidden_channels=training_args.deep_hidden_channels,
                out_channels=training_args.deep_out_dim,
                num_layers=training_args.deep_num_layers,
                dropout=training_args.deep_dropout,
            ).to(config.device)

        predictor = LinkPredictor(
            in_channels=training_args.deep_out_dim,
            hidden_channels=training_args.deep_hidden_channels,
            out_channels=1,
            num_layers=training_args.deep_num_layers,
            dropout=training_args.deep_dropout,
        ).to(config.device)

        # setup optimizer
        params_shallow = [
            {
                "params": shallow.parameters(),
                "lr": training_args.shallow_lr,
                "weight_decay": training_args.weight_decay_shallow,
            }
        ]
        params_deep = [
            {
                "params": list(deep.parameters()) + list(predictor.parameters()),
                "lr": training_args.deep_lr,
                "weight_decay": training_args.weight_decay_deep,
            }
        ]

        optimizer_shallow = torch.optim.Adam(params_shallow)
        optimizer_deep = torch.optim.Adam(params_deep)
        criterion = nn.BCEWithLogitsLoss()
        evaluator = METRICS(
            metrics_list=config.dataset.metrics, task="LinkPrediction", dataset=config.dataset.dataset_name
        )

        # Train warm start if provided
        if training_args.warm_start > 0:
            fit_warm_start(
                warm_start=training_args.warm_start,
                shallow=shallow,
                deep=deep,
                predictor=predictor,
                data_deep=data_deep,  # data_deep if config.dataset.dataset_name == 'ogbl-collab' else data_split['train']
                data_shallow=data_shallow,
                optimizer_shallow=optimizer_shallow,
                optimizer_deep=optimizer_deep,
                criterion=criterion,
                split_edge=split_edge,
                evaluator=evaluator,
                batch_size=training_args.batch_size,
                log=log,
                config=config,
                training_args=training_args,
            )
            shallow.train()
            deep.train()
            predictor.train()

        # Now consider joint traning
        λ = nn.Parameter(torch.tensor(training_args.lambda_))
        params_combined = [
            {
                "params": shallow.parameters(),
                "lr": training_args.shallow_lr_joint,
                "weight_decay": training_args.weight_decay_shallow,
            },
            {
                "params": list(deep.parameters()) + list(predictor.parameters()),
                "lr": training_args.deep_lr_joint,
                "weight_decay": training_args.weight_decay_deep,
            },
            {"params": [λ], "lr": training_args.lambda_lr},
        ]

        optimizer = torch.optim.Adam(params_combined)

        prog_bar = tqdm(range(training_args.joint_train))

        control_model_weights = ModelWeights(
            direction=training_args.direction,
            shallow_frozen_epochs=training_args.shallow_frozen_epochs,
            deep_frozen_epochs=training_args.deep_frozen_epochs,
        )

        for epoch in prog_bar:
            shallow.train()
            deep.train()
            predictor.train()

            control_model_weights.step(epoch=epoch, shallow=shallow, deep=deep, predictor=predictor)

            pos_edge_index = data_shallow
            pos_edge_index = pos_edge_index.to(config.device)

            loss_list = []
            for perm in DataLoader(range(pos_edge_index.size(1)), training_args.batch_size, shuffle=True):
                optimizer.zero_grad()
                W = deep(
                    data_deep.x, data_deep.adj_t
                )  ##if config.dataset.dataset_name == 'ogbl-collab' else deep(data_deep.x, data_deep.edge_index)

                edge = pos_edge_index[:, perm]

                # shallow
                pos_out_shallow = shallow(edge[0], edge[1])

                # deep
                pos_out_deep = predictor(W[edge[0]], W[edge[1]])

                # Negative edges
                neg_edge_index = get_negative_samples(edge, data_deep.x.shape[0], edge.size(1))
                neg_edge_index = neg_edge_index.to(data_deep.x.device)
                neg_out_shallow = shallow(neg_edge_index[0], neg_edge_index[1])
                neg_out_deep = predictor(W[neg_edge_index[0]], W[neg_edge_index[1]])

                # Now Predictions
                pos_logits = predict(
                    shallow_logit=pos_out_shallow,
                    lambda_=λ,
                    deep_logit=pos_out_deep.squeeze(),
                    training_args=training_args,
                )
                # total negative
                neg_logits = predict(
                    shallow_logit=neg_out_shallow,
                    lambda_=λ,
                    deep_logit=neg_out_deep.squeeze(),
                    training_args=training_args,
                )

                # concat positive and negative predictions
                total_predictions = torch.cat([pos_logits, neg_logits], dim=0)
                y = torch.cat(
                    [torch.ones(pos_out_shallow.size(0)), torch.zeros(neg_out_shallow.size(0))], dim=0
                ).to(config.device)

                # calculate loss
                loss = criterion(total_predictions, y)
                loss_list.append(loss.item())
                # optimization step
                loss.backward()
                torch.nn.utils.clip_grad_norm_(deep.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(shallow.parameters(), 1.0)
                optimizer.step()

            results = test_joint_with_predictor(
                shallow=shallow,
                deep=deep,
                predictor=predictor,
                split_edge=split_edge,
                x=data_deep.x,
                adj_t=data_deep.adj_t,
                evaluator=evaluator,
                batch_size=training_args.batch_size,
                λ=λ,
                training_args=training_args,
            )
            prog_bar.set_postfix(
                {
                    "loss": loss.item(),
                    "λ": λ.item(),
                    f"Train {config.dataset.track_metric}": results["train"][config.dataset.track_metric],
                    f"Val {config.dataset.track_metric}": results["val"][config.dataset.track_metric],
                    f"Test {config.dataset.track_metric}": results["test"][config.dataset.track_metric],
                }
            )
            Logger.add_to_run(loss=np.mean(loss_list), results=results)

        Logger.end_run()
    Logger.save_results(save_path + "/combined_comb2_results.json")
    if "save_to_folder" in config:
        create_path(config.save_to_folder)
        additional_save_path = f"{config.save_to_folder}/{config.dataset.task}/{config.dataset.dataset_name}/{config.dataset.DIM}/{config.model_type}"
        create_path(f"{additional_save_path}")
        Logger.save_results(additional_save_path + f"/results_comb2_{training_args.deep_model}.json")
    Logger.get_statistics(metrics=prepare_metric_cols(config.dataset.metrics))
