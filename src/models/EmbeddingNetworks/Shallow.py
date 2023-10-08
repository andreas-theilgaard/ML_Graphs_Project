import torch
from torch_geometric.nn import Node2Vec as N2V
from tqdm import tqdm


class Node2Vec:
    def __init__(self, device, embedding_dim, save_path):
        self.device = device
        self.embedding_dim = embedding_dim
        self.save_path = save_path

    def save_embeddings(self, model):
        torch.save(
            model.embedding.weight.data.cpu(),
            f"models/{self.embedding_save_path}/embeddings.pt",
        )

    def Node2Vec(
        self,
        edge_index,
        walk_length: int = 80,
        walks_per_node: int = 10,
        context_size: int = 20,
        num_negative_samples: int = 1,
        sparse=True,
    ):
        model = Node2Vec(
            edge_index=edge_index,
            embedding_dim=self.embedding_dim,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=num_negative_samples,
            sparse=sparse,
        )
        return model

    def train(self, model, batch_size, epochs, lr, num_workers=0):
        loader = model.loader(
            batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=lr)
        model.train()
        for epoch in tqdm(range(epochs)):
            for i, (pos_sample, neg_sample) in enumerate(tqdm(loader)):
                optimizer.zero_grad()
                loss = model.loss(
                    pos_sample.to(self.device), neg_sample.to(self.device)
                )
                loss.backward()
                optimizer.step()
            # Save embedding
        self.save_embeddings(model=model)
        print(
            f"Embeddings have been saved at {self.embedding_save_path} you can now use them for any downstream task"
        )
        return model


# class ShallowEmbeddings:
#     def __init__(self,shallow_embedding: str,device: str,embedding_save_path:str ,embedding_dim : int=128):
#         self.shallow_embedding = shallow_embedding
#         self.device = device
#         self.embedding_dim = embedding_dim
#         self.embedding_save_path=embedding_save_path

#         # Get data

#     def verify_shallow_model(self):
#         assert self.shallow_embedding in ['Node2Vec'], f"Expect shallow_embedding to be of the form ['Node2Vec'], but received {self.shallow_embedding}"

#     def save_embeddings(self,model):
#         torch.save(model.embedding.weight.data.cpu(),f"models/{self.embedding_save_path}/embeddings.pt")

#     def get_model(self,data=None,walk_length: int=80, walks_per_node: int=10, context_size: int=20,num_negative_samples: int=1):
#         if self.shallow_embedding=='Node2Vec':
#             return self.Node2Vec(data.edge_index,walk_length,walks_per_node,context_size,num_negative_samples)

#     def Node2Vec(self,edge_index,walk_length: int=80, walks_per_node: int=10, context_size: int=20,num_negative_samples: int=1):
#         model = Node2Vec(edge_index=edge_index,embedding_dim=self.embedding_dim,walk_length=walk_length,context_size=context_size,walks_per_node=walks_per_node,num_negative_samples=num_negative_samples,sparse=True)
#         return model

#     def train_Node2Vec(self,model,batch_size,epochs,lr,num_workers=0):
#         loader = model.loader(batch_size=batch_size,shuffle=True,num_workers=num_workers)
#         optimizer = torch.optim.SparseAdam(list(model.parameters()),lr=lr)
#         model.train()
#         for epoch in tqdm(range(epochs)):
#             for i, (pos_sample,neg_sample) in enumerate(tqdm(loader)):
#                 optimizer.zero_grad()
#                 loss = model.loss(pos_sample.to(self.device),neg_sample.to(self.device))
#                 loss.backward()
#                 optimizer.step()
#             # Save embedding
#             self.save_embeddings(model=model)
#         return model

#     def train(self,odmel,batch_size=256,epochs=5,lr=0.01,num_workers=0):
#         if self.shallow_embedding == 'Node2Vec':
#             self.train_Node2Vec(model=model,batch_size=batch_size,epochs=epochs,lr=lr,num_workers=num_workers)
#             print(f"Embeddings have been saved at {self.embedding_save_path} you can now use them for any downstreamtask")
