from dataset import TKGDataset
from models import CompGCNConv, TGCN
import torch
from torch import nn
from torch.utils.data import DataLoader
from models import GLogicLayer

def get_param(shape):
    param = nn.Parameter(torch.Tensor(*shape))
    nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))  # relu
    # nn.init.xavier_normal_(param, gain=nn.init.calculate_gain(self.act))  # relu
    return param


dataset = TKGDataset('./data/ICEWS14', datatype="train")

ent_emb = get_param((dataset.n_ent, 128))
rel_emb = get_param((2*dataset.n_rel, 128))
ts_emb = get_param((dataset.n_ts, 128))

# conv = CompGCNConv(128, 128)
gcn = TGCN(dataset.n_ent, dataset.n_rel, 128, 128, 2, 0.1, nn.ReLU())
print(ent_emb.shape, rel_emb.shape, ts_emb.shape)

ent_emb, rel_emb = gcn(dataset.tkg, ent_emb, rel_emb, ts_emb)
print(ent_emb.shape, rel_emb.shape)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
logic_layer = GLogicLayer(dataset.n_ent, dataset.n_rel, 128, 128)

class Args:
    def __init__(self):
        self.sample_nodes = 15
        self.max_nodes = 30
        self.sample_method = 1
        self.sample_ratio = 0.5
        self.score_method = 'emb'
        self.loss = 'max_min'
        self.time_score = 1

args = Args()

for batch in dataloader:
    h, r, t, ts = batch
    print(h.shape, r.shape, t.shape, ts.shape)

    re, res = logic_layer(ent_emb, rel_emb, ts_emb, h, r, ts, dataset=dataset, args=args)
    breakpoint()

# conv(dataset.tkg, ent_emb, rel_emb, ts_emb)
