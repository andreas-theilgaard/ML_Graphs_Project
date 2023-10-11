from omegaconf import OmegaConf
import torch
from src.data.get_data import DataLoader
from torch_geometric.utils import to_undirected, negative_sampling
import random
from hydra.experimental import compose, initialize
from torch.optim import SGD
from tqdm import tqdm

# Initialize a Hydra context
with initialize(config_path="src/config"):
    config = compose(config_name="base.yaml")

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


#####################################################
# Shallow model begin here
#####################################################


def metrics(pred, label, type="accuracy"):
    if type == "accuracy":
        yhat = (pred > 0.5).float()
        return torch.mean((yhat == label).float()).item()


data = dataset[0]
if data.is_directed():
    data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)

pos_edge_index = data.edge_index
neg_edge_index = negative_sampling(
    pos_edge_index,
    num_nodes=data.num_nodes,
    num_neg_samples=int(pos_edge_index.shape[1] * training_args.num_negative_samples),
)

train_edge = torch.cat([pos_edge_index, neg_edge_index], dim=1)
train_label = torch.cat(
    [torch.ones(pos_edge_index.size(1)), torch.zeros(neg_edge_index.size(1))], dim=0
)

# Create embedding
emb_weight = (
    torch.rand((data.x.shape[0], training_args.embedding_dim))
    if training_args.initialize == "Random"
    else data.x
)
embeddings = torch.nn.Embedding(
    num_embeddings=data.x.shape[0],
    embedding_dim=training_args.embedding_dim,
    _weight=emb_weight,
)

beta = torch.nn.Parameter(torch.tensor([1.0]))
optimizer = SGD(
    [{"params": embeddings.parameters()}, {"params": [beta]}],
    lr=training_args.lr,
    momentum=training_args.lr,
)

prog_bar = tqdm(range(training_args.epochs))
loss_fn = torch.nn.BCELoss()

for i in prog_bar:
    optimizer.zero_grad()

    # get node embeddings
    z_i = embeddings(train_edge[0])
    z_j = embeddings(train_edge[1])

    difference = z_i - z_j
    norm_difference = torch.norm(difference, dim=1)
    loss_func = beta - norm_difference
    sig = torch.sigmoid(loss_func)
    loss = loss_fn(sig, train_label)
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    prog_bar.set_postfix(
        {"Loss": loss.item(), "Accuracy": metrics(pred=sig, label=train_label)}
    )


torch.save(embeddings.state_dict(), "emb.pth")


import torch

X = torch.load("21-25-04embedding.pth", map_location="cpu")

X.device
