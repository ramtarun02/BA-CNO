import torch 
import torch.nn as nn
import torch.nn.functional as F



class PointEncoder(nn.module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), )

    def forward(self, x):
        return self.net(x)


def compute_kNN(pos, k):
    # pos: (B, N, coord_dim) - (B, N, 3)
    # returns neighbour index: (B, N, k)

    dist = torch.cdist(pos, pos)
    B, N, _ = dist.shape

    eye = torch.eye(N, device=pos.device).bool()
    dist.masked_fill_(eye.unsqueeze(0), floar("inf"))

    neighbour_idx = dist.topk(k, dim=-1, largest=False).indices

    return neighbour_idx


class ResidualBlocks(nn.module):
    def __init__(self, hidden_dim):
        super()__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim),
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)
        

    def forward(self, x, neighbour_idx):

        
        B, N, H = x.shape
        k = neighbour_idx.shape[-1]
        

        idx_expanded = neighbor_idx.unsqueeze(-1).expand(B, N, k, H)
        neighbors = torch.gather(
            x.unsqueeze(1).expand(B, N, N, H),
            2,
            idx_expanded
        )                                          # (B, N, k, H)

        center = x.unsqueeze(2).expand(B, N, k, H) # (B, N, k, H)
        msg_input = torch.cat([center, neighbors], dim=-1)
        messages = self.message_mlp(msg_input)     # (B, N, k, H)

        aggregated = messages.mean(dim=2)          # (B, N, H)

        update_input = torch.cat([x, aggregated], dim=-1)
        delta = self.update_mlp(update_input)      # (B, N, H)

        out = x + delta
        out = self.norm(out)
        out = F.relu(out)
        return out

