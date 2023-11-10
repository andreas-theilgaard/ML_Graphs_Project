import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from tqdm import tqdm
from src.models.utils import set_seed, get_negative_samples
from src.models.combined.combined_link.combined_link_utils import (
    SAGE_Direct,
    SAGE,
    test_indi,
    test_joint,
    decode,
    get_split_edge,
    predict,
    ModelWeights,
    GCN,
    GCN_Direct,
)
from src.models.shallow import ShallowModel, initialize_embeddings
from src.models.metrics import METRICS
from src.models.utils import prepare_metric_cols
from torch.utils.data import DataLoader


@torch.no_grad()
def test_link_citation(
    shallow, deep, data_deep, direct, gamma, config, split_edge, training_args, evaluator, lambda_
):
    shallow.eval()
    deep.eval()

    W = deep(data_deep.x, data_deep.adj_t)

    def test_split(split):
        source = split_edge[split]["source_node"].to(config.device)
        target = split_edge[split]["target_node"].to(config.device)
        target_neg = split_edge[split]["target_node_neg"].to(config.device)

        pos_preds = []
        for perm in DataLoader(range(source.size(0)), training_args.batch_size):
            src, dst = source[perm], target[perm]
            out_shallow = shallow(src, dst)
            out_deep = (
                deep(data_deep.x, data_deep.adj_t, src, dst)
                if direct
                else decode(W=W, node_i=src, node_j=dst, gamma=gamma, type_=training_args.deep_decode)
            )
            pred = torch.sigmoid(
                predict(
                    shallow_logit=out_shallow,
                    deep_logit=out_deep,
                    lambda_=lambda_,
                    training_args=training_args,
                )
            )
            pos_preds += [pred.squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)

        neg_preds = []
        source = source.view(-1, 1).repeat(1, 1000).view(-1)
        target_neg = target_neg.view(-1)

        for perm in DataLoader(range(source.size(0)), training_args.batch_size):
            src, dst_neg = source[perm], target_neg[perm]
            out_shallow = shallow(src, dst_neg)
            out_deep = (
                deep(data_deep.x, data_deep.adj_t, src, dst_neg)
                if direct
                else decode(W=W, node_i=src, node_j=dst_neg, gamma=gamma, type_=training_args.deep_decode)
            )
            pred = torch.sigmoid(
                predict(
                    shallow_logit=out_shallow,
                    deep_logit=out_deep,
                    lambda_=lambda_,
                    training_args=training_args,
                )
            )
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
    shallow, deep, data_deep, direct, gamma, config, split_edge, training_args, evaluator
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
            out_deep = (
                deep(data_deep.x, data_deep.adj_t, src, dst)
                if direct
                else decode(W=W, node_i=src, node_j=dst, gamma=gamma, type_=training_args.deep_decode)
            )
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
            out_deep = (
                deep(data_deep.x, data_deep.adj_t, src, dst_neg)
                if direct
                else decode(W=W, node_i=src, node_j=dst_neg, gamma=gamma, type_=training_args.deep_decode)
            )
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
    data_deep,
    optimizer_shallow,
    optimizer_deep,
    data_shallow,
    criterion,
    direct,
    gamma,
    config,
    training_args,
):
    shallow.train()
    deep.train()
    optimizer_shallow.zero_grad()
    optimizer_deep.zero_grad()

    pos_edge_index = data_shallow
    pos_edge_index = pos_edge_index.to(data_deep.x.device)

    #######################
    # positive edges
    #######################
    pos_out_shallow = shallow(pos_edge_index[0], pos_edge_index[1])
    if direct:
        pos_out_deep = deep(data_deep.x, data_deep.adj_t, pos_edge_index[0], pos_edge_index[1])
    else:
        W = deep(data_deep.x, data_deep.adj_t)
        pos_out_deep = decode(
            W=W,
            node_i=pos_edge_index[0],
            node_j=pos_edge_index[1],
            gamma=gamma,
            type_=training_args.deep_decode,
        )

    #######################
    # negative edges
    #######################
    if config.dataset.dataset_name == "ogbl-citation2":
        dst_neg = torch.randint(
            0, data_deep.x.shape[0], pos_edge_index[0].size(), dtype=torch.long, device=config.device
        )
        neg_out_shallow = shallow(pos_edge_index[0], dst_neg)

    else:
        neg_edge_index = get_negative_samples(pos_edge_index, data_deep.x.shape[0], pos_edge_index.size(1))
        neg_edge_index = neg_edge_index.to(data_deep.x.device)
        # Negative edges
        neg_out_shallow = shallow(neg_edge_index[0], neg_edge_index[1])

        if direct:
            # Negative edges
            neg_out_deep = deep(data_deep.x, data_deep.adj_t, neg_edge_index[0], neg_edge_index[1])
        else:
            # Negative edges
            neg_out_deep = decode(
                W=W,
                node_i=neg_edge_index[0],
                node_j=neg_edge_index[1],
                gamma=gamma,
                type_=training_args.deep_decode,
            )

    # concat positive and negative predictions
    total_predictions_shallow = torch.cat([pos_out_shallow, neg_out_shallow], dim=0)
    total_predictions_deep = torch.cat([pos_out_deep, neg_out_deep], dim=0)
    y_shallow = torch.cat(
        [torch.ones(pos_out_shallow.size(0)), torch.zeros(neg_out_shallow.size(0))], dim=0
    ).to(data_deep.x.device)
    y_deep = torch.cat([torch.ones(pos_out_deep.size(0)), torch.zeros(neg_out_deep.size(0))], dim=0).to(
        data_deep.x.device
    )

    # calculate loss
    loss_shallow = criterion(total_predictions_shallow, y_shallow)
    loss_deep = criterion(total_predictions_deep, y_deep)

    # optimization step
    loss_shallow.backward()
    optimizer_shallow.step()

    # optimization step
    loss_deep.backward()
    optimizer_deep.step()
    return (loss_shallow, loss_deep)


def fit_warm_start(
    warm_start,
    shallow,
    deep,
    data_deep,
    data_shallow,
    optimizer_shallow,
    optimizer_deep,
    criterion,
    split_edge,
    evaluator,
    batch_size,
    direct,
    gamma,
    log,
    config,
    training_args,
):
    prog_bar = tqdm(range(warm_start))
    for i in prog_bar:
        loss_shallow, loss_deep = warm_train(
            shallow=shallow,
            deep=deep,
            data_deep=data_deep,
            data_shallow=data_shallow,
            optimizer_shallow=optimizer_shallow,
            optimizer_deep=optimizer_deep,
            criterion=criterion,
            direct=direct,
            gamma=gamma,
            config=config,
            training_args=training_args,
        )

        prog_bar.set_postfix({"Shallow L": loss_shallow.item(), "Deep L": loss_deep.item()})

        if i % 10 == 0:
            results_shallow, results_deep = (
                test_indi(
                    shallow=shallow,
                    deep=deep,
                    split_edge=split_edge,
                    x=data_deep.x,
                    adj_t=data_deep.adj_t,
                    evaluator=evaluator,
                    batch_size=batch_size,
                    direct=direct,
                )
                if config.dataset.dataset_name != "ogbl-citation2"
                else test_link_citation_indi(
                    shallow=shallow,
                    data_deep=data_deep,
                    deep=deep,
                    direct=direct,
                    gamma=gamma,
                    config=config,
                    split_edge=split_edge,
                    training_args=training_args,
                )
            )
            log.info(
                f"Shallow: Train {config.dataset.track_metric}:{results_shallow['train'][config.dataset.track_metric]}, Val {config.dataset.track_metric}:{results_shallow['val'][config.dataset.track_metric]}, Test {config.dataset.track_metric}:{results_shallow['test'][config.dataset.track_metric]}"
            )
            log.info(
                f"Deep: Train {config.dataset.track_metric}:{results_deep['train'][config.dataset.track_metric]}, Val {config.dataset.track_metric}:{results_deep['val'][config.dataset.track_metric]}, Test {config.dataset.track_metric}:{results_deep['test'][config.dataset.track_metric]}"
            )


def fit_combined1_link(config, dataset, training_args, Logger, log, seeds, save_path):
    gamma = None
    data = dataset[0]
    if data.is_directed():
        data.edge_index = to_undirected(data.edge_index)
    data = data.to(config.device)

    data_shallow, split_edge = get_split_edge(
        data=data, dataset=dataset, config=config, training_args=training_args
    )

    if "edge_weight" in data:
        data.edge_weight = data.edge_weight.view(-1).to(torch.float)
    data_deep = T.ToSparseTensor()(data)
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
        if training_args.direct:
            if training_args.deep_model == "GraphSage":
                deep = SAGE_Direct(
                    in_channels=data.num_features,
                    hidden_channels=training_args.deep_hidden_channels,
                    out_channels=training_args.deep_out_dim,
                    num_layers=training_args.deep_num_layers,
                    dropout=training_args.deep_dropout,
                    gamma=training_args.gamma,
                    decode_type=training_args.deep_decode,
                ).to(config.device)
            elif training_args.deep_model == "GCN":
                deep = GCN_Direct(
                    in_channels=data.num_features,
                    hidden_channels=training_args.deep_hidden_channels,
                    out_channels=training_args.deep_out_dim,
                    num_layers=training_args.deep_num_layers,
                    dropout=training_args.deep_dropout,
                    gamma=training_args.gamma,
                    decode_type=training_args.deep_decode,
                ).to(config.device)

        else:
            if training_args.deep_model == "GraphSage":
                deep = SAGE(
                    n_channels=data.num_features,
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

        # setup optimizer
        params_shallow = [{"params": shallow.parameters(), "lr": training_args.shallow_lr}]
        if training_args.direct:
            params_deep = [{"params": list(deep.parameters()), "lr": training_args.deep_lr}]
        else:
            gamma = nn.Parameter(torch.tensor(training_args.gamma))
            params_deep = (
                [{"params": deep.parameters() + [gamma], "lr": training_args.deep_lr}]
                if training_args.deep_decode == "dist"
                else [{"params": deep.parameters(), "lr": training_args.deep_lr}]
            )

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
                data_deep=data_deep,
                data_shallow=data_shallow,
                optimizer_shallow=optimizer_shallow,
                optimizer_deep=optimizer_deep,
                criterion=criterion,
                split_edge=split_edge,
                evaluator=evaluator,
                batch_size=training_args.batch_size,
                gamma=training_args.gamma,
                direct=training_args.direct,
                log=log,
                config=config,
                training_args=training_args,
            )
            shallow.train()
            deep.train()

        # Now consider joint traning
        λ = nn.Parameter(torch.tensor(training_args.lambda_))

        if training_args.direct:
            DEEP_PARAMS = {"params": deep.parameters(), "lr": training_args.deep_lr_joint}
        else:
            DEEP_PARAMS = (
                {"params": deep.parameters() + [gamma], "lr": training_args.deep_lr_joint}
                if training_args.deep_decode == "dist"
                else {"params": deep.parameters(), "lr": training_args.deep_lr_joint}
            )

        params_combined = [
            {"params": shallow.parameters(), "lr": training_args.shallow_lr_joint},
            DEEP_PARAMS,
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
            control_model_weights.step(epoch=epoch, shallow=shallow, deep=deep)

            optimizer.zero_grad()

            pos_edge_index = data_shallow
            pos_edge_index = pos_edge_index.to(data_deep.x.device)

            #######################
            # positive edges
            #######################
            pos_out_shallow = shallow(pos_edge_index[0], pos_edge_index[1])
            if training_args.direct:
                pos_out_deep = deep(data_deep.x, data_deep.adj_t, pos_edge_index[0], pos_edge_index[1])
            else:
                W = deep(data_deep.x, data_deep.adj_t)
                pos_out_deep = decode(
                    W=W,
                    node_i=pos_edge_index[0],
                    node_j=pos_edge_index[1],
                    gamma=gamma,
                    type_=training_args.deep_decode,
                )

            #######################
            # negative edges
            #######################
            if config.dataset.dataset_name == "ogbl-citation2":
                dst_neg = torch.randint(
                    0, data_deep.x.shape[0], pos_edge_index[0].size(), dtype=torch.long, device=config.device
                )
                neg_out_shallow = shallow(pos_edge_index[0], dst_neg)

            else:
                neg_edge_index = get_negative_samples(
                    pos_edge_index, data_deep.x.shape[0], pos_edge_index.size(1)
                )
                neg_edge_index = neg_edge_index.to(data_deep.x.device)
                # Negative edges
                neg_out_shallow = shallow(neg_edge_index[0], neg_edge_index[1])

                if training_args.direct:
                    # Negative edges
                    neg_out_deep = deep(data_deep.x, data_deep.adj_t, neg_edge_index[0], neg_edge_index[1])
                else:
                    # Negative edges
                    neg_out_deep = decode(
                        W=W,
                        node_i=neg_edge_index[0],
                        node_j=neg_edge_index[1],
                        gamma=gamma,
                        type_=training_args.deep_decode,
                    )

            # # Combine model predictions

            # total positive
            pos_logits = predict(
                shallow_logit=pos_out_shallow, deep_logit=pos_out_deep, lambda_=λ, training_args=training_args
            )
            # total negative
            neg_logits = predict(
                shallow_logit=neg_out_shallow, deep_logit=neg_out_deep, lambda_=λ, training_args=training_args
            )

            # concat positive and negative predictions
            total_predictions = torch.cat([pos_logits, neg_logits], dim=0)
            y = torch.cat(
                [torch.ones(pos_out_shallow.size(0)), torch.zeros(neg_out_shallow.size(0))], dim=0
            ).to(config.device)

            # calculate loss
            loss = criterion(total_predictions, y)

            # optimization step
            loss.backward()

            torch.nn.utils.clip_grad_norm_(shallow.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(deep.parameters(), 1.0)
            optimizer.step()

            results = (
                test_joint(
                    shallow=shallow,
                    deep=deep,
                    split_edge=split_edge,
                    x=data_deep.x,
                    adj_t=data_deep.adj_t,
                    evaluator=evaluator,
                    batch_size=training_args.batch_size,
                    λ=λ,
                    direct=training_args.direct,
                    gamma=gamma,
                    training_args=training_args,
                )
                if config.dataset.dataset_name != "ogbl-citation2"
                else test_link_citation(
                    shallow=shallow,
                    deep=deep,
                    data_deep=data_deep,
                    direct=training_args.direct,
                    split_edge=split_edge,
                    evaluator=evaluator,
                    lambda_=λ,
                )
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
            Logger.add_to_run(loss=loss.item(), results=results)

    Logger.end_run()
    Logger.save_results(save_path + "/combined_comb1_results.json")
    Logger.get_statistics(metrics=prepare_metric_cols(config.dataset.metrics))
