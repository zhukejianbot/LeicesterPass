import torch
from torch import nn
from torch.utils.data import Dataset

class PassSequenceDataset(Dataset):
    def __init__(self, X_num, X_pid, X_rid, y):
        self.X_num = torch.from_numpy(X_num).float()
        self.X_pid = torch.from_numpy(X_pid).long()
        self.X_rid = torch.from_numpy(X_rid).long()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.X_num[idx],
            self.X_pid[idx],
            self.X_rid[idx],
            self.y[idx]
        )


class PassTransformer(nn.Module):
    def __init__(self,
                 num_numeric_features: int,
                 num_players: int,
                 seq_len: int = 6,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 256,
                 dropout: float = 0.2,
                 player_emb_dim: int = 16):
        super().__init__()
        self.seq_len = seq_len

        # embeddings
        self.player_emb = nn.Embedding(num_players, player_emb_dim)
        self.receiver_emb = nn.Embedding(num_players, player_emb_dim)

        # project numeric + embeddings into d_model
        self.num_proj = nn.Linear(num_numeric_features, d_model)
        self.player_proj = nn.Linear(player_emb_dim, d_model)
        self.receiver_proj = nn.Linear(player_emb_dim, d_model)

        # positional
        self.pos_embedding = nn.Embedding(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, x_num, x_pid, x_rid, return_h: bool = False):
        """
        x_num: [B, L, F_num]
        x_pid: [B, L]
        x_rid: [B, L]
        """
        B, L, F = x_num.shape
        assert L == self.seq_len

        h_num = self.num_proj(x_num)          # [B, L, d]
        p_emb = self.player_emb(x_pid)       # [B, L, emb]
        r_emb = self.receiver_emb(x_rid)     # [B, L, emb]

        h = h_num + self.player_proj(p_emb) + self.receiver_proj(r_emb)

        positions = torch.arange(L, device=x_num.device).unsqueeze(0).expand(B, L)
        h = h + self.pos_embedding(positions)  # [B, L, d]

        h_enc = self.encoder(h)               # [B, L, d]
        h_last = h_enc[:, -1, :]              # [B, d]
        logits = self.fc(h_last).squeeze(-1)  # [B]

        if return_h:
            return logits, h_enc
        return logits