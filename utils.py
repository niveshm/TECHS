

import json


def load_json(file_path):
    with open(file_path, encoding='utf8') as f:
        json_ = json.load(f)
    return json_

def load_fact(file_path):
    facts = []
    with open(file_path, encoding='utf8') as f:
        for line in f:
            facts.append(line.strip().split('\t'))
    return facts

class Args:
    def __init__(self):
        self.sample_nodes = 600
        self.max_nodes = 100
        self.sample_method = 3
        self.sample_ratio = 0.5
        self.score_method = 'att'
        self.loss = 'bce'
        self.time_score = 0
        self.dataset = 'icews14'
        self.batch_size = 32
        self.epoch = 3
        self.lr = 0.001
        self.weight_decay = 0.0000
        self.act = 'relu'
        self.gcn_dim = 128
        self.hidden_dim = 64
        self.gcn_layer = 2
        self.logic_layer = 3
        self.gcn_drop = 0.1
        self.hidden_drop = 0.1
        self.logic_reduce = 'sum'
        self.use_gcn = 1
        self.label_smooth = 0.1
        self.logic_ratio = 0.8
        self.num_workers = 8
        self.seed = 95
