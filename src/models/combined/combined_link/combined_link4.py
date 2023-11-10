import torch
import torch.nn as nn
from src.data.get_data import DataLoader as DL
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models.utils import set_seed, get_negative_samples
from src.models.combined.combined_link.combined_link_utils import (
    SAGE_Direct,
    SAGE,
    test_indi,
    test_joint,
    decode,
    predict,
    get_split_edge,
)
from src.models.shallow import ShallowModel, initialize_embeddings
from src.models.metrics import METRICS
from src.models.utils import prepare_metric_cols
import numpy as np


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
    batch_size=65538,
    training_args=None,
):
    shallow.train()
    deep.train()

    pos_edge_index = data_shallow
    pos_edge_index = pos_edge_index.to(data_deep.x.device)

    for perm in DataLoader(range(pos_edge_index.size(1)), batch_size, shuffle=True):
        optimizer_shallow.zero_grad()
        optimizer_deep.zero_grad()

        edge = pos_edge_index[:, perm]
        pos_out_shallow = shallow(edge[0], edge[1])

        if direct:
            pos_out_deep = deep(data_deep.x, data_deep.adj_t, edge[0], edge[1])
        else:
            W = deep(data_deep.x, data_deep.adj_t)
            pos_out_deep = decode(
                W=W, node_i=edge[0], node_j=edge[1], gamma=gamma, type_=training_args.deep_decode
            )

        # Negative edges
        neg_edge_index = get_negative_samples(edge, data_deep.x.shape[0], edge.size(1))
        neg_edge_index = neg_edge_index.to(data_deep.x.device)
        neg_out_shallow = shallow(neg_edge_index[0], neg_edge_index[1])
        if direct:
            neg_out_deep = deep(data_deep.x, data_deep.adj_t, edge[0], edge[1])
        else:
            neg_out_deep = decode(
                W=W, node_i=edge[0], node_j=edge[1], gamma=gamma, type_=training_args.deep_decode
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
            training_args=training_args,
        )

        prog_bar.set_postfix({"Shallow L": loss_shallow.item(), "Deep L": loss_deep.item()})

        if i % 10 == 0:
            results_shallow, results_deep = test_indi(
                shallow=shallow,
                deep=deep,
                split_edge=split_edge,
                x=data_deep.x,
                adj_t=data_deep.adj_t,
                evaluator=evaluator,
                batch_size=batch_size,
                direct=direct,
            )
            log.info(
                f"Shallow: Train {config.dataset.track_metric}:{results_shallow['train'][config.dataset.track_metric]}, Val {config.dataset.track_metric}:{results_shallow['val'][config.dataset.track_metric]}, Test {config.dataset.track_metric}:{results_shallow['test'][config.dataset.track_metric]}"
            )
            log.info(
                f"Deep: Train {config.dataset.track_metric}:{results_deep['train'][config.dataset.track_metric]}, Val {config.dataset.track_metric}:{results_deep['val'][config.dataset.track_metric]}, Test {config.dataset.track_metric}:{results_deep['test'][config.dataset.track_metric]}"
            )


def fit_combined4_link(config, dataset, training_args, Logger, log, seeds, save_path):
    gamma = None
    if data.is_directed():
        data.edge_index = to_undirected(data.edge_index)

    data = data.to(config.device)
    data_shallow, split_edge = get_split_edge(data=data, dataset=dataset, config=config)

    if "edge_weight" in data:
        data.edge_weight = data.edge_weight.view(-1).to(torch.float)
    data_deep = T.ToSparseTensor()(data)
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
            deep = SAGE_Direct(
                in_channels=data.num_features,
                hidden_channels=training_args.deep_hidden_channels,
                out_channels=training_args.deep_out_dim,
                num_layers=training_args.deep_num_layers,
                dropout=training_args.deep_dropout,
                gamma=training_args.gamma,
                decode_type=training_args.deep_decode,
            ).to(config.device)
        else:
            deep = SAGE(
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
                [{"params": deep.parameters() + [gamma], "lr": training_args.deep_lr_joint}]
                if training_args.deep_decode == "dist"
                else [{"params": deep.parameters(), "lr": training_args.deep_lr_joint}]
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
        λ = nn.Parameter(torch.tensor(1.0))

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
        for i in prog_bar:
            shallow.train()
            deep.train()

            pos_edge_index = data_shallow
            pos_edge_index = pos_edge_index.to(config.device)
            loss_list = []
            for perm in DataLoader(range(pos_edge_index.size(1)), 65538, shuffle=True):
                optimizer.zero_grad()

                edge = pos_edge_index[:, perm]
                pos_out_shallow = shallow(edge[0], edge[1])

                if training_args.direct:
                    pos_out_deep = deep(data_deep.x, data_deep.adj_t, edge[0], edge[1])
                else:
                    W = deep(data_deep.x, data_deep.adj_t)
                    pos_out_deep = decode(
                        W=W, node_i=edge[0], node_j=edge[1], gamma=gamma, type_=training_args.deep_decode
                    )

                # Negative edges
                neg_edge_index = get_negative_samples(edge, data_deep.x.shape[0], edge.size(1))
                neg_edge_index = neg_edge_index.to(data_deep.device)
                neg_out_shallow = shallow(neg_edge_index[0], neg_edge_index[1])
                if training_args.direct:
                    neg_out_deep = deep(data_deep.x, data_deep.adj_t, edge[0], edge[1])
                else:
                    neg_out_deep = decode(
                        W=W, node_i=edge[0], node_j=edge[1], gamma=gamma, type_=training_args.deep_decode
                    )

                # total positive
                pos_logits = predict(
                    shallow_logit=pos_out_shallow,
                    deep_logit=pos_out_deep,
                    lambda_=λ,
                    training_args=training_args,
                )
                # total negative
                neg_logits = predict(
                    shallow_logit=neg_out_shallow,
                    deep_logit=neg_out_deep,
                    lambda_=λ,
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
                optimizer.step()

            results = test_joint(
                shallow=shallow,
                deep=deep,
                split_edge=split_edge,
                x=data_deep.x,
                adj_t=data_deep.adj_t,
                evaluator=evaluator,
                batch_size=training_args.batch_size,
                λ=λ,
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
    Logger.save_results(save_path + "/combined_comb1_results.json")
    Logger.get_statistics(metrics=prepare_metric_cols(config.dataset.metrics))


# import torch
# import torch.nn as nn
# from src.data.get_data import DataLoader as DL
# from torch_geometric.nn import SAGEConv
# import torch.nn.functional as F
# import torch_geometric.transforms as T
# from torch_geometric.utils import to_undirected, negative_sampling
# from ogb.linkproppred import Evaluator
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from src.models.utils import set_seed,get_seeds
# import pandas as pd
# from src.models.combined.combined_link_utils import SAGE_Direct,SAGE,get_negative_samples,test_indi,test_joint,deep_L2
# from src.models.shallow import ShallowModel
# from src.data.data_utils import get_link_data_split
# from src.models.metrics import METRICS
# from src.models.logger import LoggerClass
# from src.models.utils import prepare_metric_cols
# import logging

# metrics = ['hits@10','hits@50','hits@100','acc','auc']
# track_metric = 'hits@50'
# log = logging.getLogger(__name__)

# SHALLOW_EMB_DIM = 64
# SHALLOW_LR = 0.1
# SHALLOW_INIT = 'random'
# SHALLOW_DECODER = 'dist'
# BETA=0.0
# SHALLOW_LR_COMBINED = 0.1

# DEEP_LR = 0.005
# DIRECT = True
# DEEP_HIDDEN_CHANNELS = 256
# DEEP_OUT_DIM = 256
# DEEP_NUM_LAYERS = 3
# DEEP_DROPOUT = 0.1
# BATCH_SIZE = 65536
# DEEP_LR_COMBINED = 0.005
# LAMBDA_LR = 0.01
# GAMMA = 0.0
# DEVICE = 'cpu'
# dataset_name='ogbl-collab'


# def warm_train(shallow,deep,data_deep,optimizer_shallow,optimizer_deep,data_shallow,criterion,direct,gamma,batch_size=65538):
#     shallow.train()
#     deep.train()

#     pos_edge_index = data_shallow
#     pos_edge_index = pos_edge_index.to(data_deep.device)

#     total_loss = total_examples = 0
#     for perm in DataLoader(range(pos_edge_index.size(1)),batch_size,shuffle=True):
#         optimizer_shallow.zero_grad()
#         optimizer_deep.zero_grad()

#         edge = pos_edge_index[:,perm]
#         pos_out_shallow = shallow(edge[0], edge[1])

#         if direct:
#             pos_out_deep = deep(data_deep.x,data_deep.adj_t,edge[0],edge[1])
#         else:
#             W = deep(data_deep.x,data_deep.adj_t)
#             pos_out_deep = deep_L2(W=W,node_i=edge[0],node_j=edge[1],gamma=gamma)

#         # Negative edges
#         neg_edge_index = get_negative_samples(edge, data_deep.x.shape[0], edge.size(1))
#         neg_edge_index=neg_edge_index.to(data_deep.device)
#         neg_out_shallow = shallow(neg_edge_index[0], neg_edge_index[1])
#         if direct:
#             neg_out_deep = deep(data_deep.x,data_deep.adj_t,edge[0],edge[1])
#         else:
#             neg_out_deep = deep_L2(W=W,node_i=edge[0],node_j=edge[1],gamma=gamma)


#         # concat positive and negative predictions
#         total_predictions_shallow = torch.cat([pos_out_shallow,neg_out_shallow],dim=0)
#         total_predictions_deep = torch.cat([pos_out_deep,neg_out_deep],dim=0)
#         y_shallow = torch.cat([torch.ones(pos_out_shallow.size(0)), torch.zeros(neg_out_shallow.size(0))], dim=0).to(data_deep.device)
#         y_deep = torch.cat([torch.ones(pos_out_deep.size(0)), torch.zeros(neg_out_deep.size(0))], dim=0).to(data_deep.device)

#         # calculate loss
#         loss_shallow = criterion(total_predictions_shallow,y_shallow)
#         loss_deep = criterion(total_predictions_deep,y_deep)

#         # optimization step
#         loss_shallow.backward()
#         optimizer_shallow.step()

#         # optimization step
#         loss_deep.backward()
#         optimizer_deep.step()

#     return (loss_shallow,loss_deep)


# def fit_warm_start(warm_start,shallow,deep,data_deep,optimizer_shallow,optimizer_deep,criterion,split_edge,evaluator,batch_size,direct,gamma):
#     prog_bar = tqdm(range(warm_start))
#     for i in prog_bar:
#         loss_shallow,loss_deep=warm_train(shallow=shallow,deep=deep,data_deep=data_deep,optimizer_shallow=optimizer_shallow,optimizer_deep=optimizer_deep,criterion=criterion,direct=direct,gamma=gamma,batch_size=BATCH_SIZE)

#         prog_bar.set_postfix({'Shallow L':loss_shallow.item(),'Deep L':loss_deep.item()})

#         if i % 10 == 0:
#             results_shallow,results_deep=test_indi(shallow=shallow,deep=deep,split_edge=split_edge,x=data_deep.x,adj_t=data_deep.adj_t,evaluator=evaluator,batch_size=batch_size)
#             print(f"Shallow: Train {track_metric}:{results_shallow['train'][track_metric]}, Val {track_metric}:{results_shallow['val'][track_metric]}, Test {track_metric}:{results_shallow['test'][track_metric]}")
#             print(f"Deep: Train {track_metric}:{results_deep['train'][track_metric]}, Val {track_metric}:{results_deep['val'][track_metric]}, Test {track_metric}:{results_deep['test'][track_metric]}")


# def fit_combined4_link(dataset_name,device='cpu'):
#     gamma = None
#     dataset = DL(
#         task_type="LinkPrediction",
#         dataset=dataset_name,
#         model_name="Shallow",
#     ).get_data()

#     data = dataset[0]
#     if data.is_directed():
#         data.edge_index = to_undirected(data.edge_index)

#     Logger = LoggerClass(
#         runs=len(seeds),
#         metrics=prepare_metric_cols(metrics=metrics),
#         seeds=seeds,
#         log=log,
#     )
#     seeds = get_seeds(n=3)

#     for counter,seed in enumerate(seeds):
#         set_seed(seed)
#         Logger.start_run()
#         warm_start = 100
#         joint_train = 400


#         ##### Setup models #####
#         shallow = ShallowModel(num_nodes = data.x.shape[0], embedding_dim=SHALLOW_EMB_DIM,beta=BETA,init_embeddings=SHALLOW_INIT,decoder_type=SHALLOW_DECODER,device=device).to(device)
#         if DIRECT:
#             deep = SAGE_Direct(in_channels=data.num_features, hidden_channels=DEEP_HIDDEN_CHANNELS, out_dim=DEEP_OUT_DIM, num_layers=DEEP_NUM_LAYERS, dropout=DEEP_DROPOUT,gamma=GAMMA).to(device)
#         else:
#             deep = SAGE(in_channels=data.num_features, hidden_channels=DEEP_HIDDEN_CHANNELS, out_dim=DEEP_OUT_DIM, num_layers=DEEP_NUM_LAYERS, dropout=DEEP_DROPOUT).to(device)


#         # setup optimizer
#         params_shallow =  [{'params': shallow.parameters(), 'lr': SHALLOW_LR}]
#         if DIRECT:
#             params_deep =[{'params': list(deep.parameters()), 'lr': DEEP_LR}]
#         else:
#             gamma = nn.Parameter(torch.tensor(GAMMA))
#             params_deep =[{'params': list(deep.parameters()) + [gamma], 'lr': DEEP_LR}]


#         optimizer_shallow = torch.optim.Adam(params_shallow)
#         optimizer_deep = torch.optim.Adam(params_deep)
#         criterion = nn.BCEWithLogitsLoss()

#         if dataset_name in ["ogbl-collab", "ogbl-ppi"]:
#             split_edge = dataset.get_edge_split()
#         elif dataset_name in ["Cora", "Flickr"]:
#             train_data, val_data, test_data = get_link_data_split(data)
#             edge_weight_in = data.edge_weight if "edge_weight" in data else None
#             edge_weight_in = edge_weight_in.float() if edge_weight_in else edge_weight_in
#             split_edge = {
#                 "train": {"edge": train_data.pos_edge_label_index.T, "weight": edge_weight_in},
#                 "valid": {
#                     "edge": val_data.pos_edge_label_index.T,
#                     "edge_neg": val_data.neg_edge_label_index.T,
#                 },
#                 "test": {
#                     "edge": test_data.pos_edge_label_index.T,
#                     "edge_neg": test_data.neg_edge_label_index.T,
#                 },
#             }
#         data_shallow = (split_edge["train"]["edge"]).T

#         if 'edge_weight' in data:
#             data.edge_weight = data.edge_weight.view(-1).to(torch.float)

#         data_deep = T.ToSparseTensor()(data)
#         data_deep = data_deep.to(device)


#         evaluator = METRICS(metrics_list=metrics, task='LinkPrediction')


#         # Train warm start if provided
#         if warm_start>0:
#             fit_warm_start(warm_start=warm_start,shallow=shallow,deep=deep,data_deep=data_deep,optimizer_shallow=optimizer_shallow,optimizer_deep=optimizer_deep,criterion=criterion,split_edge=split_edge,evaluator=evaluator,batch_size=BATCH_SIZE,gamma=gamma,direct=DIRECT)
#             shallow.train()
#             deep.train()


#         # Now consider joint traning
#         λ = nn.Parameter(torch.tensor(1.0))

#         if DIRECT:
#             DEEP_PARAMS = {'params': deep.parameters(), 'lr': DEEP_LR_COMBINED}
#         else:
#             DEEP_PARAMS = {'params': deep.parameters()+[gamma], 'lr': DEEP_LR_COMBINED}

#         params_combined = [
#                 {'params': shallow.parameters(), 'lr': SHALLOW_LR_COMBINED},
#                 DEEP_PARAMS,
#                 {'params': [λ], 'lr': LAMBDA_LR}
#             ]

#         optimizer = torch.optim.Adam(params_combined)

#         prog_bar = tqdm(range(joint_train))
#         for i in prog_bar:
#             shallow.train()
#             deep.train()


#             pos_edge_index = data_shallow
#             pos_edge_index = pos_edge_index.to(device)
#             total_loss = total_examples = 0
#             for perm in DataLoader(range(pos_edge_index.size(1)),65538,shuffle=True):
#                 optimizer.zero_grad()

#                 edge = pos_edge_index[:,perm]
#                 pos_out_shallow = shallow(edge[0], edge[1])

#                 if DIRECT:
#                     pos_out_deep = deep(data_deep.x,data_deep.adj_t,edge[0],edge[1])
#                 else:
#                     W = deep(data_deep.x,data_deep.adj_t)
#                     pos_out_deep = deep_L2(W=W,node_i=edge[0],node_j=edge[1],gamma=gamma)

#                 # Negative edges
#                 neg_edge_index = get_negative_samples(edge, data_deep.x.shape[0], edge.size(1))
#                 neg_edge_index=neg_edge_index.to(data_deep.device)
#                 neg_out_shallow = shallow(neg_edge_index[0], neg_edge_index[1])
#                 if DIRECT:
#                     neg_out_deep = deep(data_deep.x,data_deep.adj_t,edge[0],edge[1])
#                 else:
#                     neg_out_deep = deep_L2(W=W,node_i=edge[0],node_j=edge[1],gamma=gamma)

#                 # total positive
#                 pos_logits = pos_out_shallow + λ * (pos_out_deep)
#                 # total negative
#                 neg_logits = neg_out_shallow + λ * (neg_out_deep)

#                 # concat positive and negative predictions
#                 total_predictions = torch.cat([pos_logits,neg_logits],dim=0)
#                 y = torch.cat([torch.ones(pos_out_shallow.size(0)), torch.zeros(neg_out_shallow.size(0))], dim=0).to(device)

#                 # calculate loss
#                 loss = criterion(total_predictions,y)

#                 # optimization step
#                 loss.backward()
#                 optimizer.step()
#                 prog_bar.set_postfix({'Shallow L':loss.item(),'λ':λ.item()})

#                 results=test_joint(shallow=shallow,deep=deep,split_edge=split_edge,x=data_deep.x,adj_t=data_deep.adj_t,evaluator=evaluator,batch_size=65536,λ=λ)
#                 Logger.add_to_run(loss=loss, results=results)

#             if i % 10 == 0:
#                 print(f"Train {track_metric}:{results['train'][track_metric]}, Val {track_metric}:{results['val'][track_metric]}, Test {track_metric}:{results['test'][track_metric]}")

#     Logger.end_run()
#     Logger.save_results('combined_test' + f"/results_combined4_{warm_start}_{DEEP_LR}_{DEEP_LR_COMBINED}_{SHALLOW_LR}_{SHALLOW_INIT}_{DEEP_HIDDEN_CHANNELS}_{DEEP_OUT_DIM}_{SHALLOW_EMB_DIM}_{DIRECT}.json")
#     Logger.get_statistics(metrics=prepare_metric_cols(metrics))


# if __name__ == "__main__":
#     fit_combined4_link(dataset_name=dataset_name,device=DEVICE)
