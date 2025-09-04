import random
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from dataset import TKGDataset
from model import TLoGN
from utils import Args
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import os
from os.path import join
import time

def save_model(path, model, args):
    state = {
        'model': model.state_dict(),
        'args': vars(args)
    }
    torch.save(state, path)


def gen_result_str(results):
    return 'MR:{}  MRR:{}  Hits@1:{}  Hits@3:{}  Hits@10:{}'.format(results['MR'], results['MRR'], results['Hits@1'], results['Hits@3'], results['Hits@10'])

class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.1, reduction='mean'):  # 2 0.25
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        # pt = torch.sigmoid(predict) # sigmoide获取概率
        pt = predict
        #在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) \
               - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

class Runner():
    def __init__(self, args: Args):
        self.data_dir = f'./data/{args.dataset}'

        self.train_data = TKGDataset(self.data_dir, datatype="train", logic=True, logic_ratio=args.logic_ratio)
        self.valid_data = TKGDataset(self.data_dir, datatype="valid")
        self.test_data = TKGDataset(self.data_dir, datatype="test")

        self.result_dir = "./results"
        if os.path.exists(self.result_dir)==False:
            os.makedirs(self.result_dir)
        



        self.train_dataloader = DataLoader(self.train_data, batch_size=args.batch_size, shuffle=True)
        self.valid_dataloader = DataLoader(self.valid_data, batch_size=args.batch_size, shuffle=False)
        self.test_dataloader = DataLoader(self.test_data, batch_size=args.batch_size, shuffle=False)

        self.args = args
        self.n_ent = self.train_data.n_ent
        self.n_rel = self.train_data.n_rel
        self.n_ts = self.train_data.n_ts
        self.label_smooth = 1 - args.label_smooth


        self.act = torch.relu if args.act == 'relu' else torch.tanh

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using device:", self.device)

        # Move dataset tensors to device for efficiency
        self.train_data.train_facts_tensor = self.train_data.train_facts_tensor.to(self.device)
        self.valid_data.train_facts_tensor = self.valid_data.train_facts_tensor.to(self.device)
        self.test_data.train_facts_tensor = self.test_data.train_facts_tensor.to(self.device)

        self.tkg = self.train_data.tkg.to(self.device)

        self.model = TLoGN(self.n_ent, self.n_rel, self.n_ts, act=self.act, device=self.device, args=args).to(self.device)

        total_params = sum([param.nelement() for param in self.model.parameters()])
        print('Model Parameters Num:', total_params)
        for name, param in self.model.named_parameters():
            print(name, param.size(), param.device, str(param.requires_grad))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if args.loss == 'bce':
            self.loss = nn.BCELoss()
        if args.loss == 'focal':
            self.loss = BCEFocalLoss()
    
    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def gen_target_one_hot(self, target_ent, pre_ents):
        '''
        :param target_ent: B
        :param pre_ents: X*2
        :return:
        '''
        one_hot_label = torch.from_numpy(
            np.array([int(ent == target_ent[batch_id]) for batch_id, ent in pre_ents], dtype=np.float32)).to(self.device)
        # smooth
        one_hot_label = one_hot_label * self.label_smooth + 0.0001
        return one_hot_label
    
    def train_epoch(self):
        self.model.train()
        losses = []
        for batch in tqdm(self.train_dataloader):
            h, r, t, ts = batch
            query_head = h.to(self.device)
            query_rel = r.to(self.device)
            query_tail = t.to(self.device)
            query_ts = ts.to(self.device)

            if self.args.use_gcn == 1:
                ent_emd, rel_emd, time_emd = self.model.gen_gcn_emd(self.tkg)
            else:
                ent_emd, rel_emd, time_emd = self.model.gen_gcn_emd(None)
            
            tail_ents, tail_scores = self.model.logic_layer(ent_emd, rel_emd, time_emd, query_head, query_rel, query_ts, dataset=self.train_data, args=self.args, device=self.device)

            if self.args.loss == 'max_min':
                batch_size = h.size()[0]
                scores_all = torch.zeros((batch_size, self.n_ent)).to(self.device)
                scores_all[[tail_ents[:, 0], tail_ents[:, 1]]] = tail_scores
                tail_scores = scores_all
                pos_scores = tail_scores[[torch.arange(batch_size).to(self.device), query_tail]]
                max_n = torch.max(tail_scores, 1, keepdim=True)[0]
                loss = torch.sum(- pos_scores + max_n + torch.log(torch.sum(torch.exp(tail_scores - max_n), 1)))
            else:
                target_one_hot = self.gen_target_one_hot(query_tail, tail_ents)  # X
                loss = self.calc_loss(tail_scores, target_one_hot)

            self.optimizer.zero_grad()
            loss.backward()

            clip_grad_norm_(self.model.parameters(), max_norm=5)
            self.optimizer.step()
            losses.append(loss.item())

        loss = np.mean(losses)
        return loss

    def eval_epoch(self, datatype="valid"):
        self.model.eval()
        
        if datatype == "valid":
            dataloader = self.valid_dataloader
            dataset = self.valid_data
        else:
            dataloader = self.test_dataloader
            dataset = self.test_data
        
        ranking = []
        results = dict()
        for batch in tqdm(dataloader):
            h, r, t, ts, y = batch
            query_head = h.to(self.device)
            query_rel = r.to(self.device)
            query_tail = t.to(self.device)
            query_ts = ts.to(self.device)
            labels = y.to(self.device)

            if self.args.use_gcn == 1:
                ent_emd, rel_emd, time_emd = self.model.gen_gcn_emd(self.tkg)
            else:
                ent_emd, rel_emd, time_emd = self.model.gen_gcn_emd(None)

            tail_ents, tail_scores = self.model.logic_layer(ent_emd, rel_emd, time_emd, query_head, query_rel, query_ts, dataset=dataset, args=self.args, device=self.device)

            batch_size = y.size()[0]

            scores_all = torch.zeros((batch_size, self.n_ent)).to(self.device)
            scores_all[[tail_ents[:, 0], tail_ents[:, 1]]] = tail_scores
            pred = scores_all

            obj = query_tail

            b_range = torch.arange(batch_size).to(self.device)
            target_pred = pred[b_range, obj]

            pred = torch.where(labels.bool(), -1e8, pred)
            pred[b_range, obj] = target_pred

            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]
            ranks = ranks.float()

            results['count'] = torch.numel(ranks) + results.get('count', 0)  # number of predictions
            results['MR'] = torch.sum(ranks).item() + results.get('MR', 0)
            results['MRR'] = torch.sum(1.0 / ranks).item() + results.get('MRR', 0)

            for k in [1, 3, 10]:
                results[f'Hits@{k}'] = torch.numel(ranks[ranks <= k]) + results.get(f'Hits@{k}', 0)
        

        count = results['count']
        for key_ in results.keys():
            results[key_] = round(results[key_] / count, 5)
        return results
            



    def train(self):
        valid_metric = 'Hits@1'
        time_str = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))
        save_dir = join(self.result_dir, self.args.dataset + '_logic_{}'.format(time_str))
        if os.path.exists(save_dir)==False:
            os.makedirs(save_dir)
        
        best_metric = -1.0
        

        for epoch in range(self.args.epoch):
            train_loss = self.train_epoch()
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}")
        
            results = self.eval_epoch(datatype="valid")
            results_str = gen_result_str(results)
            print(f"Epoch {epoch}: Valid Results: {results_str}")
            if results[valid_metric] > best_metric:
                save_model(join(save_dir, 'train_model.pt'), runner.model, args)
                best_metric = results[valid_metric]
        

        # logging.info('==============Testing================')
        state = torch.load(join(save_dir, 'train_model.pt'))
        self.model.load_state_dict(state['model'])
        results = self.eval_epoch(datatype="test")
        results_str = gen_result_str(results)
        print(results_str)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    args = Args()
    set_seed(args.seed)
    runner = Runner(args)
    runner.train()

