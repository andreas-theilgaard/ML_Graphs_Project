import torch
import torch.nn as nn
from src.data.get_data import DataLoader as DL
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models.utils import set_seed, get_seeds, get_negative_samples
from src.models.combined.combined_link.combined_link_utils import (
    LinkPredictor,
    SAGE,
    test_indi_with_predictor,
    test_joint_with_emb_combined,
    get_split_edge,
    ModelWeights,
    GCN,
)
from src.models.shallow import ShallowModel, initialize_embeddings
from src.models.metrics import METRICS
from src.models.utils import prepare_metric_cols
import numpy as np


@torch.no_grad()
def test_link_citation(
    shallow, deep, predictor, data_deep, config, split_edge, training_args, evaluator, lambda_
):
    shallow.eval()
    deep.eval()
    predictor.eval()

    W = deep(data_deep.x, data_deep.adj_t)

    def test_split(split):
        source = split_edge[split]["source_node"].to(config.device)
        target = split_edge[split]["target_node"].to(config.device)
        target_neg = split_edge[split]["target_node_neg"].to(config.device)

        pos_preds = []
        for perm in DataLoader(range(source.size(0)), training_args.batch_size):
            src, dst = source[perm], target[perm]
            out_shallow = shallow(src, dst)
            out_deep = predictor(W[src], W[dst])
            pred = torch.sigmoid(out_shallow + lambda_ * out_deep)
            pos_preds += [pred.squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)

        neg_preds = []
        source = source.view(-1, 1).repeat(1, 1000).view(-1)
        target_neg = target_neg.view(-1)

        for perm in DataLoader(range(source.size(0)), training_args.batch_size):
            src, dst_neg = source[perm], target_neg[perm]
            out_shallow = shallow(src, dst_neg)
            out_deep = predictor(W[src], W[dst_neg])
            pred = torch.sigmoid(out_shallow + lambda_ * out_deep)
            neg_preds += [pred.squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)
        return {"pos": pos_pred, "neg": neg_pred}

    train = test_split("eval_train")
    valid = test_split("valid")
    test = test_split("test")

    predictions = {
        "train": {"y_pred_pos": train["pos"], "y_pred_neg": train["neg"]},
        "val": {"y_pred_pos": valid["pos"], "y_pred_neg": valid["neg"]},
        "test": {"y_pred_pos": test["pos"], "y_pred_neg": test["neg"]},
    }
    results = evaluator.collect_metrics(predictions)
    return results


@torch.no_grad()
def test_link_citation_indi(
    shallow, deep, predictor, data_deep, config, split_edge, training_args, evaluator
):
    shallow.eval()
    deep.eval()

    W = deep(data_deep.x, data_deep.adj_t)

    def test_split(split):
        source = split_edge[split]["source_node"].to(config.device)
        target = split_edge[split]["target_node"].to(config.device)
        target_neg = split_edge[split]["target_node_neg"].to(config.device)

        pos_preds_shallow = []
        pos_preds_deep = []
        for perm in DataLoader(range(source.size(0)), training_args.batch_size):
            src, dst = source[perm], target[perm]
            pos_preds_shallow += [torch.sigmoid(shallow(src, dst)).squeeze().cpu()]
            out_deep = predictor(W[src], W[dst])
            pos_preds_deep += [torch.sigmoid(out_deep).squeeze().cpu()]

        pos_pred_shallow = torch.cat(pos_preds_shallow, dim=0)
        pos_pred_deep = torch.cat(pos_preds_deep, dim=0)

        neg_preds_shallow = []
        neg_preds_deep = []

        source = source.view(-1, 1).repeat(1, 1000).view(-1)
        target_neg = target_neg.view(-1)

        for perm in DataLoader(range(source.size(0)), training_args.batch_size):
            src, dst_neg = source[perm], target_neg[perm]
            neg_preds_shallow += [torch.sigmoid(shallow(src, dst_neg)).squeeze().cpu()]
            out_deep = predictor(W[src], W[dst_neg])
            neg_preds_deep += [torch.sigmoid(out_deep).squeeze().cpu()]
        neg_pred_shallow = torch.cat(neg_preds_shallow, dim=0).view(-1, 1000)
        neg_pred_deep = torch.cat(neg_preds_deep, dim=0).view(-1, 1000)

        return {
            "pos_shallow": pos_pred_shallow,
            "neg_shallow": neg_pred_shallow,
            "pos_deep": pos_pred_deep,
            "neg_deep": neg_pred_deep,
        }

    train = test_split("eval_train")
    valid = test_split("valid")
    test = test_split("test")

    predictions_shallow = {
        "train": {"y_pred_pos": train["pos_shallow"], "y_pred_neg": train["neg_shallow"]},
        "val": {"y_pred_pos": valid["pos_shallow"], "y_pred_neg": valid["neg_shallow"]},
        "test": {"y_pred_pos": test["pos_shallow"], "y_pred_neg": test["neg_shallow"]},
    }
    predictions_deep = {
        "train": {"y_pred_pos": train["pos_deep"], "y_pred_neg": train["neg_deep"]},
        "val": {"y_pred_pos": valid["pos_deep"], "y_pred_neg": valid["neg_deep"]},
        "test": {"y_pred_pos": test["pos_deep"], "y_pred_neg": test["neg_deep"]},
    }

    results_shallow = evaluator.collect_metrics(predictions_shallow)
    results_deep = evaluator.collect_metrics(predictions_deep)
    return (results_shallow, results_deep)


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

        W = deep(data_deep.x, data_deep.adj_t)

        edge = pos_edge_index[:, perm]

        # shallow
        pos_out_shallow = shallow(edge[0], edge[1])

        # deep
        pos_out_deep = predictor(W[edge[0]], W[edge[1]])

        # Negative edges
        if config.dataset.dataset_name == "ogbl-citation2":
            dst_neg = torch.randint(
                0, data_deep.x.shape[0], edge[0].size(), dtype=torch.long, device=config.device
            )
            neg_out_shallow = shallow(edge[0], dst_neg)
            neg_out_deep = predictor(W[edge[0]], W[dst_neg])
        else:
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
    log,
    config,
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
            results_shallow, results_deep = (
                test_indi_with_predictor(
                    shallow=shallow,
                    deep=deep,
                    linkpredictor=predictor,
                    split_edge=split_edge,
                    x=data_deep.x,
                    adj_t=data_deep.adj_t,
                    evaluator=evaluator,
                    batch_size=batch_size,
                )
                if config.dataset.dataset_name != "ogbl-citation2"
                else test_link_citation_indi(
                    shallow=shallow,
                    deep=deep,
                    predictor=predictor,
                    data_deep=data_deep,
                    config=config,
                    split_edge=split_edge,
                    training_args=training_args,
                    evaluator=evaluator,
                )
            )
            log.info(
                f"Shallow: Train {config.dataset.track_metric}:{results_shallow['train'][config.dataset.track_metric]}, Val {config.dataset.track_metric}:{results_shallow['val'][config.dataset.track_metric]}, Test {config.dataset.track_metric}:{results_shallow['test'][config.dataset.track_metric]}"
            )
            log.info(
                f"Deep: Train {config.dataset.track_metric}:{results_deep['train'][config.dataset.track_metric]}, Val {config.dataset.track_metric}:{results_deep['val'][config.dataset.track_metric]}, Test {config.dataset.track_metric}:{results_deep['test'][config.dataset.track_metric]}"
            )


def fit_combined3_link(config, dataset, training_args, Logger, log, seeds, save_path):
    data = dataset[0]
    undirected = True
    if data.is_directed():
        undirected = False
        data.edge_index = to_undirected(data.edge_index)
    data = data.to(config.device)

    data_shallow, split_edge = get_split_edge(
        data=data, dataset=dataset, config=config, training_args=training_args
    )

    if "edge_weight" in data:
        data.edge_weight = data.edge_weight.view(-1).to(torch.float)
    data_deep = T.ToSparseTensor()(data)
    if not undirected:
        data_deep.adj_t = data_deep.adj_t.to_symmetric()
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
                n_channels=data.num_features,
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
        params_shallow = [{"params": shallow.parameters(), "lr": training_args.shallow_lr}]
        params_deep = [
            {"params": list(deep.parameters()) + list(predictor.parameters()), "lr": training_args.deep_lr}
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
                data_deep=data_deep,
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
        predictor = LinkPredictor(
            in_channels=training_args.deep_out_dim + training_args.embedding_dim,
            hidden_channels=training_args.deep_hidden_channels,
            out_channels=1,
            num_layers=training_args.deep_num_layers,
            dropout=training_args.deep_dropout,
        ).to(config.device)

        params_combined = [
            {"params": shallow.parameters(), "lr": training_args.shallow_lr_joint},
            {
                "params": list(deep.parameters()) + list(predictor.parameters()),
                "lr": training_args.deep_lr_joint,
            },
        ]

        optimizer = torch.optim.Adam(params_combined)

        prog_bar = tqdm(range(training_args.joint_train))
        control_model_weights = ModelWeights(
            direction=training_args.direction,
            shallow_frozen_epochs=training_args.shallow_frozen_epochs,
            deep_frozen_epochs=training_args.deep_frozen_epochs,
        )
        shallow_embeddings = (shallow.embeddings).to(config.device)

        for epoch in prog_bar:
            shallow_embeddings.train()
            deep.train()
            predictor.train()
            control_model_weights.step(epoch=epoch, shallow=shallow_embeddings, deep=deep)

            pos_edge_index = data_shallow
            pos_edge_index = pos_edge_index.to(config.device)

            loss_list = []
            for perm in DataLoader(range(pos_edge_index.size(1)), training_args.batch_size, shuffle=True):
                optimizer.zero_grad()

                # get Embeedings from shallow and deep
                W = deep(data_deep.x, data_deep.adj_t)
                Z = shallow_embeddings.weight.data.to(config.device)
                concat_embeddings = torch.cat([Z, W], dim=-1).to(config.device)

                # positive edges
                edge = pos_edge_index[:, perm]
                pos_out = predictor(concat_embeddings[edge[0]], concat_embeddings[edge[1]])

                # Negative edges
                if config.dataset.dataset_name == "ogbl-citation2":
                    dst_neg = torch.randint(
                        0, data_deep.x.shape[0], edge[0].size(), dtype=torch.long, device=config.device
                    )
                    neg_out = predictor(concat_embeddings[edge[0], dst_neg])
                else:
                    neg_edge_index = get_negative_samples(edge, data_deep.x.shape[0], edge.size(1))
                    neg_edge_index = neg_edge_index.to(data_deep.x.device)
                    neg_out = predictor(
                        concat_embeddings[neg_edge_index[0]], concat_embeddings[neg_edge_index[1]]
                    )

                # concat positive and negative predictions
                total_predictions = torch.cat([pos_out, neg_out], dim=0)
                y = (
                    torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))], dim=0)
                    .to(config.device)
                    .unsqueeze_(1)
                )

                # calculate loss
                loss = criterion(total_predictions, y)
                loss_list.append(loss.item())

                # optimization step
                loss.backward()
                torch.nn.utils.clip_grad_norm_(deep.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(shallow_embeddings.parameters(), 1.0)
                optimizer.step()

            results = test_joint_with_emb_combined(
                shallow=shallow_embeddings,
                deep=deep,
                predictor=predictor,
                split_edge=split_edge,
                x=data_deep.x,
                adj_t=data_deep.adj_t,
                evaluator=evaluator,
                batch_size=training_args.batch_size,
            )
            prog_bar.set_postfix(
                {
                    "loss": loss.item(),
                    f"Train {config.dataset.track_metric}": results["train"][config.dataset.track_metric],
                    f"Val {config.dataset.track_metric}": results["val"][config.dataset.track_metric],
                    f"Test {config.dataset.track_metric}": results["test"][config.dataset.track_metric],
                }
            )
            Logger.add_to_run(loss=np.mean(loss_list), results=results)

    Logger.end_run()
    Logger.save_results(save_path + "/combined_comb3_results.json")
    Logger.get_statistics(metrics=prepare_metric_cols(config.dataset.metrics))
