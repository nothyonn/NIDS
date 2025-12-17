# src/AutoEncoder/model.py

import torch
import torch.nn as nn


class TCNBlock(nn.Module):
    """
    기존 TCNBlock이랑 거의 동일.
    입력/출력: [B, C, L]
    """
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu2 = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        # 길이 맞춰서 crop
        if out.size(-1) > x.size(-1):
            out = out[..., : x.size(-1)]

        return self.relu2(out + x)


class TCNAutoencoder(nn.Module):
    """
    FlowWindowDataset용 시퀀스 AutoEncoder

    입력:
      - numeric: [B, L, F_num]
      - cat:     [B, L, 3] (sport_idx, dport_idx, proto_idx)
      - mask:    [B, L]

    출력:
      - recon_numeric: [B, L, F_num]
    """

    def __init__(
        self,
        numeric_dim: int,
        num_port_classes: int,
        num_proto_classes: int,
        d_model: int = 128,
        tcn_kernel_size: int = 3,
        tcn_num_layers: int = 3,
        dropout: float = 0.1,
        port_emb_dim: int = 8,
        proto_emb_dim: int = 4,
    ):
        super().__init__()

        self.numeric_dim = numeric_dim
        self.d_model = d_model

        # --- Embeddings for categorical features ---
        self.sport_emb = nn.Embedding(num_port_classes, port_emb_dim)
        self.dport_emb = nn.Embedding(num_port_classes, port_emb_dim)
        self.proto_emb = nn.Embedding(num_proto_classes, proto_emb_dim)

        total_input_dim = numeric_dim + 2 * port_emb_dim + proto_emb_dim

        # numeric + cat -> d_model
        self.input_proj = nn.Linear(total_input_dim, d_model)

        # TCN encoder/decoder (여기선 symmetrical하게 같은 블록 반복)
        self.tcn_blocks = nn.ModuleList()
        for i in range(tcn_num_layers):
            dilation = 2 ** i
            self.tcn_blocks.append(
                TCNBlock(
                    channels=d_model,
                    kernel_size=tcn_kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        # d_model -> numeric_dim 복원
        self.output_proj = nn.Linear(d_model, numeric_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, numeric, cat, mask):
        """
        numeric: [B, L, F_num]
        cat    : [B, L, 3]
        mask   : [B, L]
        """
        B, L, _ = numeric.shape

        # --- Categorical embedding ---
        sport_idx = cat[..., 0]   # [B, L]
        dport_idx = cat[..., 1]   # [B, L]
        proto_idx = cat[..., 2]   # [B, L]

        sport_emb = self.sport_emb(sport_idx)  # [B, L, E_port]
        dport_emb = self.dport_emb(dport_idx)  # [B, L, E_port]
        proto_emb = self.proto_emb(proto_idx)  # [B, L, E_proto]

        x = torch.cat([numeric, sport_emb, dport_emb, proto_emb], dim=-1)  # [B, L, total_input_dim]

        # --- project to d_model ---
        x = self.input_proj(x)  # [B, L, d_model]

        # --- TCN ---
        # [B, L, D] -> [B, D, L]
        x = x.transpose(1, 2)  # [B, D, L]

        for block in self.tcn_blocks:
            x = block(x)  # [B, D, L]

        x = x.transpose(1, 2)  # [B, L, D]

        # --- project back to numeric_dim ---
        recon_numeric = self.output_proj(x)  # [B, L, F_num]

        return recon_numeric


def masked_mse_loss(recon, target, mask):
    """
    recon, target: [B, L, F]
    mask        : [B, L]  (1=real, 0=pad)
    """
    # [B, L, 1]
    mask = mask.unsqueeze(-1)

    diff2 = (recon - target) ** 2  # [B, L, F]
    diff2 = diff2 * mask           # padding 위치는 0

    # 유효한 element 개수
    denom = mask.sum() * target.size(-1) + 1e-8
    loss = diff2.sum() / denom
    return loss
