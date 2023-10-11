import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from ogb.linkproppred import Evaluator
from tqdm import tqdm


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(model, predictor, data, split_edge, optimizer, batch_size):
    model.train()
    predictor.train()

    pos_train_edge = split_edge["train"]["edge"].to(data.x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()

        h = model(data.x, data.adj_t)

        edge = pos_train_edge[perm].t()

        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        edge = torch.randint(
            0, data.num_nodes, edge.size(), dtype=torch.long, device=h.device
        )
        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()

    h = model(data.x, data.adj_t)

    pos_train_edge = split_edge["train"]["edge"].to(h.device)
    pos_valid_edge = split_edge["valid"]["edge"].to(h.device)
    neg_valid_edge = split_edge["valid"]["edge_neg"].to(h.device)
    pos_test_edge = split_edge["test"]["edge"].to(h.device)
    neg_test_edge = split_edge["test"]["edge_neg"].to(h.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    h = model(data.x, data.full_adj_t)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval(
            {
                "y_pred_pos": pos_train_pred,
                "y_pred_neg": neg_valid_pred,
            }
        )[f"hits@{K}"]
        valid_hits = evaluator.eval(
            {
                "y_pred_pos": pos_valid_pred,
                "y_pred_neg": neg_valid_pred,
            }
        )[f"hits@{K}"]
        test_hits = evaluator.eval(
            {
                "y_pred_pos": pos_test_pred,
                "y_pred_neg": neg_test_pred,
            }
        )[f"hits@{K}"]

        results[f"Hits@{K}"] = (train_hits, valid_hits, test_hits)

    return results


def GNN_link_trainer(dataset, config, training_args, save_path, log):
    data = dataset[0]
    edge_index = data.edge_index
    data.edge_weight = data.edge_weight.view(-1).to(torch.float)
    data = T.ToSparseTensor()(data)
    split_edge = dataset.get_edge_split()
    evaluator = Evaluator(name=config.dataset)

    if training_args.use_valedges:
        val_edge_index = split_edge["valid"]["edge"].t()
        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        data.full_adj_t = SparseTensor.from_edge_index(full_edge_index).t()
        data.full_adj_t = data.full_adj_t.to_symmetric()
    else:
        data.full_adj_t = data.adj_t

    data = data.to(config.device)

    model = SAGE(
        in_channels=data.num_features,
        hidden_channels=training_args.hidden_channels,
        num_layers=training_args.num_layers,
        dropout=training_args.dropout,
        out_channels=training_args.hidden_channels,
    ).to(config.device)
    predictor = LinkPredictor(
        in_channels=training_args.hidden_channels,
        hidden_channels=training_args.hidden_channels,
        out_channels=1,
        num_layers=training_args.num_layers,
        dropout=training_args.dropout,
    ).to(config.device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()), lr=training_args.lr
    )
    prog_bar = tqdm(range(training_args.epochs))
    save_results = np.zeros((training_args.epochs, 4))

    for i, epoch in enumerate(prog_bar):
        loss = train(
            model=model,
            predictor=predictor,
            data=data,
            split_edge=split_edge,
            optimizer=optimizer,
            batch_size=training_args.batch_size,
        )
        results = test(
            model=model,
            predictor=predictor,
            data=data,
            split_edge=split_edge,
            evaluator=evaluator,
            batch_size=training_args.batch_size,
        )
        prog_bar.set_postfix(
            {
                "Train Loss": loss,
                "Train Acc.": results["Hits@50"][0],
                "Val Acc.": results["Hits@50"][1],
                "Test Acc.": results["Hits@50"][2],
            }
        )
        save_results[i, :] = (
            loss,
            results["Hits@50"][0],
            results["Hits@50"][1],
            results["Hits@50"][2],
        )

    np.save(save_path + "/results.npy", save_results)
    np.save(save_path + "/results.npy", save_results)
    model.train()
    torch.save(model.state_dict(), save_path + "/model.pth")
