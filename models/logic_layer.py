from torch import nn
import torch
import numpy as np
from torch_scatter import scatter

from dataset import TKGDataset

def group_max_norm(logits, group_ids, device=-1):
    length = logits.size()[0]
    N_segment = max(group_ids) + 1
    group_ids = group_ids.data.cpu().numpy()

    sparse_index = torch.LongTensor(np.vstack([group_ids, np.arange(length)]))
    sparse_value = torch.ones(length, dtype=torch.float)
    trans_matrix_sparse_th = torch.sparse_coo_tensor(sparse_index, sparse_value, torch.Size([N_segment, length]), dtype=torch.float, device=device)
    # torch.sparse_coo_tensor
    norm_den = torch.sparse.mm(trans_matrix_sparse_th, logits.unsqueeze(1)) ## aggregate over logits for each group

    sparse_index = torch.LongTensor(np.vstack([np.arange(length), group_ids]))
    sparse_value = torch.ones(length, dtype=torch.float)
    trans_matrix_sparse_th = torch.sparse_coo_tensor(sparse_index, sparse_value, torch.Size([length, N_segment]), dtype=torch.float, device=device)
    den = torch.squeeze(torch.sparse.mm(trans_matrix_sparse_th, norm_den)) ## assign to each element the aggregate value of its group
    res = logits / den
    res[res != res] = 0

    return res

class GNNModel(nn.Module):
    def __init__(self, emb_dim, gcn_dim, n_rel, act=lambda x: x, max_nodes=10, reduce='sum'):
        super(GNNModel, self).__init__()
        self.emb_dim = emb_dim
        self.gcn_dim = gcn_dim
        self.n_rel = n_rel
        self.act = act
        self.max_nodes = max_nodes
        self.reduce = reduce

        self.zero_rel_emd = self.get_param([1, gcn_dim])

        self.W_message = nn.Linear(emb_dim*2, emb_dim)
        self.att_1 = nn.Linear(3*emb_dim, 1)
        self.att_2 = nn.Linear(emb_dim, 1)
        self.W_h = nn.Linear(emb_dim+emb_dim, emb_dim)
        self.lin_e_ts = nn.Linear(gcn_dim+emb_dim, emb_dim)
        self.lin_r_ts = nn.Linear(gcn_dim, emb_dim)
        self.lin_time = nn.Linear(emb_dim, emb_dim)

    
    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def forward(self, query_time, query_emd, temp_neighbors_facts, ent_emd, rel_emd, hidden_node, hidden_rel, hidden_time, att_agg, time_emd, gru1, batch_size, device, max_nodes):
        b = temp_neighbors_facts[:, 0]  # batch index
        n = temp_neighbors_facts[:, 1]
        h = temp_neighbors_facts[:, 2]
        r = temp_neighbors_facts[:, 3]
        t = temp_neighbors_facts[:, 4]
        ts = temp_neighbors_facts[:, 5]

        hidden_node_pre = hidden_node[n] # n_j
        hidden_rel_pre = hidden_rel[n] # o_j

        rel_emd = torch.cat([self.zero_rel_emd, rel_emd], dim=0)
        h_r = rel_emd[r+1]
        h_t = ent_emd[t]
        h_ts = time_emd[ts]

        h_t = torch.cat([h_t, h_ts], dim=-1)
        h_t = self.lin_e_ts(h_t)
        h_r = self.lin_r_ts(h_r)
        h_ts = self.lin_time(h_ts) # no use

        message = torch.cat([hidden_node_pre, h_r], dim=-1)
        message = self.W_message(message)

        att_q = query_emd[b]
        att_m = message
        att_t = h_t
        att_input = torch.cat([att_q, att_m, att_t], dim=-1)
        att1 = torch.sigmoid(self.att_1(att_input))


        hidden_rel_new, _ = gru1(h_r.unsqueeze(0), hidden_rel_pre.unsqueeze(0))
        hidden_rel_new = hidden_rel_new.squeeze(0)

        hidden_fol = hidden_rel_new
        att2 = torch.sigmoid(self.att_2(hidden_fol))

        att1 = att1.squeeze(1)
        att2 = att2.squeeze(1)

        attention = (att1 + att2) / 2

        topk = max_nodes
        new_index = None
        for i in range(batch_size):
            batch_index = b == i
            temp_att = attention[batch_index]
            temp_index = torch.nonzero(batch_index).squeeze(1)

            if temp_att.size(0) > topk:
                topk_index = torch.topk(temp_att, topk).indices
                temp_index = temp_index[topk_index]
            
            if i == 0:
                new_index = temp_index
            else:
                new_index = torch.cat([new_index, temp_index], dim=0)
        
        temp_neighbors_facts = temp_neighbors_facts[new_index]
        message = message[new_index]
        hidden_rel_new = hidden_rel_new[new_index]

        hidden_time_new = None

        att1 = att1[new_index]
        att2 = att2[new_index]
        b = b[new_index]
        n = n[new_index]
        attention = (att1 + att2) / 2

        if att_agg != None:
            node_att_pre = att_agg[n]
            attention = attention * node_att_pre
        
        tail_nodes, tail_index = torch.unique(temp_neighbors_facts[:, [0, 4, 5]], dim=0, sorted=True, return_inverse=True)

        tail_e = ent_emd[tail_nodes[:, 1]]
        tail_ts = time_emd[tail_nodes[:, 2]]
        tail_emd = torch.cat([tail_e, tail_ts], dim=-1)
        tail_emd = self.lin_e_ts(tail_emd)

        attention = group_max_norm(attention, b, device=device)
        message = attention.unsqueeze(1) * message
        message_agg = scatter(message, index=tail_index, dim=0, reduce=self.reduce)
        message_agg = torch.cat([message_agg, tail_emd], dim=-1)
        hidden_node_new = self.act(self.W_h(message_agg))
        hidden_rel_new = scatter(attention.unsqueeze(1) * hidden_rel_new, index=tail_index, dim=0, reduce=self.reduce)
        att_agg_new = scatter(attention, index=tail_index, dim=0, reduce='sum')

        return tail_nodes, hidden_node_new, hidden_rel_new, hidden_time_new, att_agg_new




class GLogicLayer(nn.Module):
    def __init__(self, n_ent, n_rel, gcn_dim, emb_dim, n_layer=3, dropout=0.1, max_nodes=10, act=lambda x: x, reduce='sum', device=None):
        super(GLogicLayer, self).__init__()
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.gcn_dim = gcn_dim
        self.emb_dim = emb_dim
        self.n_layer = n_layer
        self.dropout = dropout
        self.max_nodes = max_nodes
        self.act = act
        self.reduce = reduce
        self.device = device

        self.gnn_layers = []
        for _ in range(self.n_layer):
            self.gnn_layers.append(GNNModel(emb_dim, gcn_dim, n_rel, act=self.act, max_nodes=self.max_nodes, reduce=self.reduce))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)

        self.dropout = nn.Dropout(dropout)
        self.gru1 = nn.GRU(self.emb_dim, self.emb_dim)

        self.line_ent = nn.Linear(gcn_dim, emb_dim)
        self.line_rel = nn.Linear(gcn_dim, emb_dim)
        self.line_time = nn.Linear(gcn_dim, emb_dim)
        self.line_query = nn.Linear(emb_dim * 3, emb_dim)
        self.W_final = nn.Linear(self.emb_dim, 1, bias=False)
    

    def forward(self, ent_emd, rel_emd, time_emd, query_head, query_rel, query_time, dataset: TKGDataset, args, device=None):
        # ent_emb: [n_ent, emb_dim]
        # rel_emb: [n_rel, emb_dim]
        # time_emb: [n_ts, emb_dim]
        # query_head: [batch_size,]
        # query_rel: [batch_size,]
        # query_time: [batch_size,]
        # dataset: TKGDataset object
        batch_size = query_head.size(0)
        sample_nodes = args.sample_nodes
        max_nodes = args.max_nodes
        sample_method = args.sample_method
        sample_ratio = args.sample_ratio
        score_method = args.score_method
        loss = args.loss

        time_emd = self.line_time(time_emd)
        query_time_emd = time_emd[query_time]
        query_head_emd = ent_emd[query_head]
        query_rel_emd = rel_emd[query_rel]

        query_head_emd = self.line_ent(query_head_emd) # batch_size x emb_dim
        query_rel_emd = self.line_rel(query_rel_emd) # batch_size x emb_dim
        query_time_emd = query_time_emd

        # print(query_head_emd.shape, query_rel_emd.shape, query_time_emd.shape) # batch_size x emb_dim

        query_emd = self.line_query(torch.cat([query_head_emd, query_rel_emd, query_time_emd], dim=-1)) 
        # print(query_emd.shape) # batch_size x emb_dim

        hidden_node = query_head_emd
        hidden_rel = query_rel_emd
        hidden_time = query_time_emd
        tail_nodes = None  # N*3
        att_agg = None

        for layer in range(self.n_layer):
            if layer == 0:
                # X*5 b,h,r,t,ts
                temp_neighbors_facts = dataset.load_neighbors4model_1(query_head, query_time, device, sample_method, sample_nodes, sample_ratio)
            else:
                temp_neighbors_facts = dataset.load_neighbors4model_2(tail_nodes, query_time, device, sample_method, sample_nodes, sample_ratio)

            tail_nodes, hidden_node, hidden_rel, hidden_time, att_agg = self.gnn_layers[layer](query_time, query_emd, temp_neighbors_facts, ent_emd, rel_emd, hidden_node, hidden_rel, hidden_time, att_agg, time_emd, self.gru1, batch_size, device, max_nodes)
            hidden_node = self.dropout(hidden_node)
            hidden_rel = self.dropout(hidden_rel)
            # hidden_time = self.dropout(hidden_time)
            query_time = query_time[tail_nodes[:, 0]] 

        if args.time_score == 1:
            time_scores = (tail_nodes[:,2] - query_time) * 0.1
            time_scores = torch.exp(time_scores)
            att_agg += time_scores
        
        if score_method == 'emd':
            score_emd = self.W_final(hidden_node).squeeze(1)
            score_emd = torch.relu(score_emd)
            att_agg = score_emd
        elif score_method == 'att':
            att_agg = att_agg
        elif score_method == 'both':
            score_emd = self.W_final(hidden_node).squeeze(1)
            score_emd = torch.relu(score_emd)
            att_agg = score_emd + att_agg
        
        tail_ents, tail_index = torch.unique(tail_nodes[:, [0, 1]], dim=0, sorted=True, return_inverse=True)  # X*2 N
        scores = scatter(att_agg, index=tail_index, dim=0, reduce=self.reduce)

        if loss == 'bce':
            scores = group_max_norm(scores, tail_ents[:, 0], device=device)
        
        return tail_ents, scores