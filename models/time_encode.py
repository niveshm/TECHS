from torch import nn
import torch
import numpy as np

class TimeEncode(nn.Module):
    def __init__(self, time_dim):
        super(TimeEncode, self).__init__()

        self.time_dim = time_dim
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())
    

    def forward(self, ts):
        # ts: [N,]
        ts = ts.unsqueeze(-1) # N*1
        map_ts = ts * self.basis_freq.view(1, -1)  # N*D
        map_ts += self.phase.view(1, -1)  # N*D
        harmonic = torch.cos(map_ts)
        return harmonic  # N*D
    


if __name__ == "__main__":
    emder1 = TimeEncode(100)
    print(emder1.basis_freq.size())
    print(emder1(torch.tensor([1,2,100])))