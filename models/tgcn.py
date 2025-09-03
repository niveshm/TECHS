import torch
from torch import nn
from models.comp_conv import CompGCNConv
from torch_geometric.data import Data

class TGCN(nn.Module):
    def __init__(self, num_ent, num_rel, input_dim, gcn_dim, n_layer, gcn_drop=0.1, act=None):
        super(TGCN, self).__init__()

        self.num_ent = num_ent
        self.num_rel = num_rel

        self.init_dim, self.gcn_dim, self.embed_dim = gcn_dim, gcn_dim, gcn_dim
        self.gcn_drop = gcn_drop
        self.act = act
        self.n_layer = n_layer

        self.conv1 = CompGCNConv(self.init_dim, self.gcn_dim)
        self.conv2 = CompGCNConv(self.gcn_dim, self.embed_dim) if n_layer == 2 else None
        self.lin_time = nn.Linear(input_dim, gcn_dim)

        self.time_ln = nn.LayerNorm(gcn_dim)
        self.ent_ln1 = nn.LayerNorm(gcn_dim)
        self.rel_ln1 = nn.LayerNorm(gcn_dim)
        self.ent_ln2 = nn.LayerNorm(gcn_dim)
        self.rel_ln2 = nn.LayerNorm(gcn_dim)

        self.drop = nn.Dropout(gcn_drop)

    def forward(self, g: Data, ent_emb, rel_emb, time_emd):
        # g: PyG object
        # ent_emb: [num_ent, init_dim]
        # rel_emb: [num_rel, init_dim]
        # time_emd: [num_ts, input_dim]

        time_emd = self.lin_time(time_emd)
        time_emd = self.time_ln(time_emd)

        x, r = ent_emb, rel_emb
        x, r = self.conv1(g, x, r, time_emd)
        x = self.ent_ln1(x)
        x = self.act(x)
        x = self.drop(x)
        r = self.rel_ln1(r)
        r = self.act(r)
        r = self.drop(r)

        if self.n_layer == 2:
            x, r = self.conv2(g, x, r, time_emd)
            x = self.ent_ln2(x)
            x = self.act(x)
            x = self.drop(x)
            r = self.rel_ln2(r)
            r = self.act(r)
            r = self.drop(r)

        return x, r