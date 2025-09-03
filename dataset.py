from collections import defaultdict
from torch.utils.data import Dataset
import numpy as np
from utils import load_json, load_fact
from os.path import join
from torch_geometric.data import Data
import torch
from torch_scatter import scatter_add

class TKGDataset(Dataset):
    def __init__(self, data_dir, datatype="train", logic=False, logic_ratio=0.5):
        self.data_dir = data_dir
        self.datatype = datatype
        self.logic = logic
        self.logic_ratio = logic_ratio


        self.entity2id = load_json(join(self.data_dir, 'entity2id.json'))
        self.relation2id = load_json(join(self.data_dir, 'relation2id.json'))
        self.ts2id = load_json(join(self.data_dir, 'ts2id.json'))

        self.n_ent = len(self.entity2id)
        self.n_rel = len(self.relation2id) # without inverse
        self.n_ts = len(self.ts2id) 

        train_facts = load_fact(join(self.data_dir, 'train.txt'))
        valid_facts = load_fact(join(self.data_dir, 'valid.txt'))
        test_facts = load_fact(join(self.data_dir, 'test.txt'))
        self.n_facts = len(train_facts)

        self.train_facts = self.map2ids(train_facts, self.entity2id, self.relation2id, self.ts2id)
        self.valid_facts = self.map2ids(valid_facts, self.entity2id, self.relation2id, self.ts2id)
        self.test_facts = self.map2ids(test_facts, self.entity2id, self.relation2id, self.ts2id)

        self.all_facts = np.concatenate([self.train_facts, self.valid_facts, self.test_facts], axis=0)
        # self.train_facts_new = None
        self.tkg = self.build_tkg(self.train_facts, self.n_ent)
        self.Dic_E = self.load_neighbor_dic()

        facts4train = self.all_facts
        self.train_facts_tensor = torch.tensor(facts4train)

        self.h_r_ts_dic_all = self.get_h_r_ts_dic_all()

        self.train_facts_new = []
        if self.logic:
            max_train_time = max(self.train_facts[:, -1])
            start_time = int(max_train_time * (1-self.logic_ratio))
            train_facts = self.train_facts
            fact_times = train_facts[:, -1]
            train_facts = train_facts[fact_times >= start_time]
            self.train_facts_new = train_facts
        
        self.dataset_describe()

    def dataset_describe(self):
        print('===========Dataset Description:===========')
        print('Entity Num:{} Relation Num:{} Time Num:{}'.format(self.n_ent, self.n_rel, self.n_ts))
        print('Train Num:{} Real Train Num:{} Valid Num:{} Test Num:{}'.format(len(self.train_facts), len(self.train_facts_new), len(self.valid_facts), len(self.test_facts)))
    
    def get_h_r_ts_dic_all(self):
        h_r_ts_dic = defaultdict(list)
        for item in self.all_facts:
            h, r, t, ts = item
            h_r_ts_dic[(h, r, ts)].append(t)
        return h_r_ts_dic
    
    def load_neighbor_dic(self):
        Dic_E = defaultdict(list)  # entity: fact列表 h_fact_dic = defaultdict(list)  # 头实体的邻居fact index
        # for i, fact in enumerate(self.train_facts):
        for i, fact in enumerate(self.all_facts):
            h, r, t, ts = fact
            Dic_E[h].append(i)
        for ent in Dic_E.keys():
            Dic_E[ent] = np.array(Dic_E[ent])
        return Dic_E


    def build_tkg(self, facts, n_ent):
        h = torch.tensor(facts[:,0], dtype=torch.long)
        r = torch.tensor(facts[:,1], dtype=torch.long)
        t = torch.tensor(facts[:,2], dtype=torch.long)
        ts = torch.tensor(facts[:,3], dtype=torch.long)

        edge_index = torch.stack([h, t], dim=0)  # [2, num_edges]
        edge_type = r  # [num_edges]
        edge_ts = ts  # [num_edges]
        x = torch.arange(n_ent, dtype=torch.long).unsqueeze(-1)  # [num_ent, 1], node features are just the node ids
        data = Data(x=x, edge_index=edge_index, edge_type=edge_type, edge_ts=edge_ts)
        return data

    def map2ids(self, facts, ent2id, rel2id, ts2id, add_rev=True):
        fact_ids = []
        for item in facts:
            # print(item)
            # print(item)
            h = ent2id[item[0]]
            r = rel2id[item[1]]
            t = ent2id[item[2]]
            ts = ts2id[item[3]]
            fact_ids.append([h, r, t, ts])
        if add_rev:
            for item in facts:
                h = ent2id[item[0]]
                r = rel2id[item[1]]
                t = ent2id[item[2]]
                ts = ts2id[item[3]]
                fact_ids.append([t, r+self.n_rel, h, ts])
        return np.array(fact_ids)
    
    def __getitem__(self, index_):
        if self.datatype == 'train':
            if self.logic:
                train_facts = self.train_facts_new
            else:
                train_facts = self.train_facts
            h,r,t,ts = train_facts[index_]
            return h, r, t, ts
        elif self.datatype == 'valid':
            h, r, t, ts = self.valid_facts[index_]
            h_r_ts = (h, r, ts)
            all_tails = self.h_r_ts_dic_all[h_r_ts]
            y = np.zeros([self.n_ent], dtype=np.long)
            y[all_tails] = 1
            return h, r, t, ts, y
        elif self.datatype == 'test':  # 目前不考虑mask
            h, r, t, ts = self.test_facts[index_]
            h_r_ts = (h, r, ts)
            all_tails = self.h_r_ts_dic_all[h_r_ts]
            y = np.zeros([self.n_ent], dtype=np.long)
            y[all_tails] = 1
            return h, r, t, ts, y
    
    def __len__(self):
        if self.datatype == 'train':
            if self.logic:
                train_facts = self.train_facts_new
            else:
                train_facts = self.train_facts
            return len(train_facts)
        elif self.datatype == 'valid':
            return len(self.valid_facts)
        elif self.datatype == 'test':
            return len(self.test_facts)
    
    def load_neighbors_by_array(self, _ent, query_time, sample_method, sample_nodes, sample_ratio, mask_2=False, pre_time=None):
        parent_indexs = []
        neighbors = []
        for i, ent in enumerate(_ent):
            tmp_time = query_time[i]
            neighbor_index = self.Dic_E[ent]
            neighbor_ts = self.all_facts[neighbor_index][:, -1]
            mask1 = neighbor_ts < tmp_time
            if sample_method == 1:
                mask_ = mask1
            elif sample_method == 2:
                if mask_2:
                    mask2 = neighbor_ts <= pre_time[i]
                    mask_ = mask1 & mask2
                else:
                    mask_ = mask1
            elif sample_method == 3:
                if mask_2:
                    mask2 = neighbor_ts >= pre_time[i]
                    mask_ = mask1 & mask2

                else:
                    mask_ = mask1

            neighbor_index = neighbor_index[mask_]
            if len(neighbor_index) > sample_nodes:
                if mask_2:
                    if sample_method == 1:
                        neighbor_ts = neighbor_ts[mask_] - pre_time[i]
                        weights = 1.0 / (np.abs(neighbor_ts)+1)
                        weights = np.power(weights, sample_ratio)
                    elif sample_method == 2:
                        neighbor_ts = neighbor_ts[mask_] - pre_time[i]
                        weights = np.exp(neighbor_ts * sample_ratio) + 1e-9
                    elif sample_method == 3:
                        neighbor_ts = neighbor_ts[mask_] - tmp_time

                        weights = np.exp(neighbor_ts * sample_ratio) + 1e-9

                else:
                    neighbor_ts = neighbor_ts[mask_] - tmp_time
                    weights = np.exp(neighbor_ts * sample_ratio) + 1e-9
                weights = weights / sum(weights)
                neighbor_index = np.random.choice(neighbor_index, sample_nodes, replace=False, p=weights)
            index1 = [i] * len(neighbor_index)
            neighbors.append(neighbor_index)
            parent_indexs.extend(index1)

        parent_indexs = np.array(parent_indexs)
        neighbors = np.concatenate(neighbors, axis=0)
        return parent_indexs, neighbors


    def load_neighbors4model_1(self, batch_data, query_time, device, sample_method, sample_nodes, sample_ratio):
        # Return B*N
        batch_data_array = batch_data.data.cpu().numpy()
        batch = batch_data.size(0)

        query_time = query_time.data.cpu().numpy()
        batch_index, nonzero_values = self.load_neighbors_by_array(batch_data_array, query_time, sample_method, sample_nodes, sample_ratio)

        batch_index = torch.LongTensor(batch_index).to(device)
        nonzero_values = torch.LongTensor(nonzero_values).to(device)

        temp_neighbors_facts = self.train_facts_tensor[nonzero_values]  # X*4 h,r,t,ts
        temp_neighbors_facts = torch.cat([batch_index.unsqueeze(1), batch_index.unsqueeze(1), temp_neighbors_facts], dim=1)  # X*6 b,h,r,t,ts

        batch_list = np.array(range(batch), dtype=np.long)
        h_list = batch_data_array
        r_list = np.ones(batch, dtype=np.long) * -1
        t_list = batch_data_array
        ts_list = np.zeros(batch, dtype=np.long)
        temp_array = np.stack([batch_list, batch_list, h_list, r_list, t_list, ts_list], axis=1)  # B*6
        temp_neighbors_facts = torch.cat([torch.tensor(temp_array).to(device), temp_neighbors_facts], dim=0)  # X*6 b,n,h,r,t,ts

        return temp_neighbors_facts # X*6 b,n,h,r,t,ts
    
    def load_neighbors4model_2(self, batch_data, query_time, device, sample_method, sample_nodes, sample_ratio):
        num_nodes, _ = batch_data.size()
        batch_data_array = batch_data.data.cpu().numpy()

        query_time = query_time.data.cpu().numpy()
        pre_time = batch_data_array[:,2]
        node_index, nonzero_values = self.load_neighbors_by_array(batch_data_array[:, 1], query_time, sample_method, sample_nodes, sample_ratio, mask_2=True, pre_time=pre_time)

        node_index = torch.LongTensor(node_index).to(device)
        nonzero_values = torch.LongTensor(nonzero_values).to(device)

        temp_neighbors_facts = self.train_facts_tensor[nonzero_values]  # X*4 h,r,t,ts

        batch_index = batch_data[:,0][node_index]
        temp_neighbors_facts = torch.cat([batch_index.unsqueeze(1), node_index.unsqueeze(1), temp_neighbors_facts], dim=1)  # X*6 b,n,h,r,t,ts

        batch_list = batch_data_array[:,0]
        node_list = np.arange(num_nodes)
        h_list = batch_data_array[:,1]
        r_list = np.ones(num_nodes, dtype=np.long) * -1
        t_list = batch_data_array[:,1]
        ts_list = pre_time
        temp_array = np.stack([batch_list, node_list, h_list, r_list, t_list, ts_list], axis=1)

        temp_neighbors_facts = torch.cat([torch.tensor(temp_array).to(device), temp_neighbors_facts], dim=0)

        return temp_neighbors_facts # X*6 b,n,h,r,t,ts
        

if __name__ == "__main__":
    
    data = TKGDataset('./data/ICEWS14', datatype="train")
    print(data[0])
    print(len(data))

    ## varify tkg check if train_facts match with edges check neighbors for one instance
    print(data.tkg)
    print(data.tkg.edge_index[:, :10])
    print(data.tkg.edge_type[:10])
    print(data.tkg.edge_ts[:10])
    print(data.train_facts[:10])
    print(data.tkg.x[:10])
    print(data.tkg.x.shape)
    print(data.n_ent, data.n_rel, data.n_ts, data.n_facts)
    print("***********")
    node_id = 0
    print("node_id:", node_id)
    print("neighbors:", data.tkg.edge_index[1][data.tkg.edge_index[0]==node_id])
    print("facts neighbors:", data.train_facts[data.train_facts[:,0]==node_id])
    print("relations:", data.tkg.edge_type[data.tkg.edge_index[0]==node_id])
    print("timestamps:", data.tkg.edge_ts[data.tkg.edge_index[0]==node_id])
    print("***********")
    ## verify scatter_add function get degree of each node
    deg = scatter_add(torch.ones_like(data.tkg.edge_index[0]), data.tkg.edge_index[0], dim=0, dim_size=data.n_ent)
    print("degree:", deg)
