from torch import nn
from models import TimeEncode, TGCN, GLogicLayer
import torch
import numpy as np

from utils import Args

class TLoGN(nn.Module):
    def __init__(self, n_ent, n_rel, n_ts, act, device, args:Args):
        super(TLoGN, self).__init__()
        
        self.input_dim = args.gcn_dim
        self.gcn_dim = args.gcn_dim
        self.hidden_dim = args.hidden_dim

        self.gcn_drop = args.gcn_drop
        self.hidden_drop = args.hidden_drop

        self.gcn_layer = args.gcn_layer
        self.logic_layer = args.logic_layer
        self.max_nodes = args.max_nodes

        self.device = device
        self.act = act

        self.use_gcn = args.use_gcn

        self.n_ts = n_ts
        self.n_ent = n_ent
        self.n_rel = n_rel

        self.time_encoder = TimeEncode(self.input_dim)
        self.gcn_layer = TGCN(n_ent, n_rel, self.input_dim, self.gcn_dim, n_layer=self.gcn_layer, gcn_drop=self.gcn_drop, act=self.act)

        self.logic_layer = GLogicLayer(n_ent, n_rel, self.gcn_dim, self.hidden_dim, n_layer=self.logic_layer, dropout=self.hidden_drop, max_nodes=self.max_nodes, act=self.act, reduce=args.logic_reduce, device=self.device)

        self.ent_emds = self.get_param([n_ent, self.gcn_dim])
        self.rel_emds = self.get_param([n_rel*2, self.gcn_dim])
        # print(self.ent_emds.shape, self.rel_emds.shape)


    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param
    
    def gen_time_emd(self):
        time = torch.arange(self.n_ts).to(self.device)
        time_emd = self.time_encoder(time)
        return time_emd

    def gen_gcn_emd(self, g):
        time_emds = self.gen_time_emd()
        if g is None:
            return self.ent_emds, self.rel_emds, time_emds
        else:
            ent_emds, rel_emds = self.gcn_layer(g, self.ent_emds, self.rel_emds, time_emds)
            ent_emds = self.ent_emds + ent_emds
            rel_emds = self.rel_emds + rel_emds
            return ent_emds, rel_emds, time_emds