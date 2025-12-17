# src/tcn_transformer/model.py

import torch
import torch.nn as nn
import math


class TCNBlock(nn.Module):
    """
    1D dilated Conv 기반 TCN 블록 (Residual)
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
        """
        x: [B, C, L]
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        # causal cut (padding 제거) – 여기서는 길이 동일하게 맞추도록 crop
        if out.size(-1) > x.size(-1):
            out = out[..., : x.size(-1)]

        return self.relu2(out + x)  # Residual
    


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding
    입력/출력: [B, L, D]
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)  # [L, D]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [L, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수

        pe = pe.unsqueeze(0)  # [1, L, D]
        self.register_buffer("pe", pe)  # 학습 X, buffer로 저장

    def forward(self, x):
        """
        x: [B, L, D]
        """
        L = x.size(1)
        return x + self.pe[:, :L, :]



class TCNTransformerModel(nn.Module):
    """
    FlowWindowDataset용 TCN + Transformer + CLS 기반 멀티라벨 분류 모델

    입력:
      - numeric: [B, L, F_num]   (77)
      - cat:     [B, L, 3]       (sport_idx, dport_idx, proto_idx)
      - mask:    [B, L]          (1=실제 flow, 0=padding)

    출력:
      - logits: [B, num_classes]
    """

    def __init__(
        self,
        numeric_dim: int,
        num_classes: int,
        num_port_classes: int,
        num_proto_classes: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        tcn_channels: int = 128,
        tcn_kernel_size: int = 3,
        tcn_num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 129,  # seq_len(128) + CLS(1)
        port_emb_dim: int = 8,
        proto_emb_dim: int = 4,
    ):
        super().__init__()

        self.numeric_dim = numeric_dim
        self.num_classes = num_classes
        self.d_model = d_model

        # --- Embeddings for categorical features ---
        self.sport_emb = nn.Embedding(num_port_classes, port_emb_dim)
        self.dport_emb = nn.Embedding(num_port_classes, port_emb_dim)
        self.proto_emb = nn.Embedding(num_proto_classes, proto_emb_dim)

        total_input_dim = numeric_dim + 2 * port_emb_dim + proto_emb_dim

        # numeric + cat -> d_model
        self.input_proj = nn.Linear(total_input_dim, d_model)

        # TCN blocks (on [B, D, L])
        tcn_blocks = []
        for i in range(tcn_num_layers):
            dilation = 2 ** i
            tcn_blocks.append(
                TCNBlock(
                    channels=tcn_channels,
                    kernel_size=tcn_kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.tcn_proj_in = nn.Linear(d_model, tcn_channels)
        self.tcn_blocks = nn.ModuleList(tcn_blocks)
        self.tcn_proj_out = nn.Linear(tcn_channels, d_model)

        # CLS 토큰 (learnable)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len)

        # Transformer Encoder (batch_first=True: [B, L, D])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, numeric, cat, mask):
        """
        numeric: [B, L, F_num]
        cat:     [B, L, 3]  (sport_idx, dport_idx, proto_idx)
        mask:    [B, L]     (1=real, 0=pad)
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
        x_tcn = self.tcn_proj_in(x)          # [B, D_tcn, L]
        x_tcn = x_tcn.transpose(1, 2)        # [B, L, D_tcn] -> [B, D_tcn, L]

        for block in self.tcn_blocks:
            x_tcn = block(x_tcn)            # [B, D_tcn, L]

        x_tcn = x_tcn.transpose(1, 2)       # [B, L, D_tcn]
        x_tcn = self.tcn_proj_out(x_tcn)    # [B, L, d_model]

        # --- Add CLS token ---
        cls_token = self.cls_token.expand(B, 1, self.d_model)  # [B, 1, D]
        x_seq = torch.cat([cls_token, x_tcn], dim=1)           # [B, L+1, D]

        # --- Positional Encoding ---
        x_seq = self.pos_encoder(x_seq)  # [B, L+1, D]

        # --- Padding mask for Transformer ---
        # mask: [B, L], 1=real, 0=pad  → key_padding_mask: True=pad
        key_padding = (mask == 0)  # [B, L] (bool)

        # CLS는 항상 real (pad 아님)
        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
        key_padding = torch.cat([cls_pad, key_padding], dim=1)  # [B, L+1]

        # --- Transformer Encoder ---
        # src_key_padding_mask: [B, L+1], True=ignore
        x_enc = self.transformer(x_seq, src_key_padding_mask=key_padding)  # [B, L+1, D]

        # --- CLS pooling ---
        cls_out = x_enc[:, 0, :]  # [B, D]

        # --- Classification ---
        cls_out = self.dropout(cls_out)
        logits = self.classifier(cls_out)  # [B, num_classes]

        return logits