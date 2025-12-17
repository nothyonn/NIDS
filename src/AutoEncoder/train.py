# src/AutoEncoder/train.py

from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader

from preprocess import FlowWindowDataset
from AutoEncoder.model import TCNAutoencoder, masked_mse_loss


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0

    for step, (numeric, cat, mask, y) in enumerate(loader, 1):
        numeric = numeric.to(device)  # [B, L, F]
        cat = cat.to(device)          # [B, L, 3]
        mask = mask.to(device)        # [B, L]

        optimizer.zero_grad()
        recon = model(numeric, cat, mask)  # [B, L, F]
        loss = masked_mse_loss(recon, numeric, mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 100 == 0:
            avg = total_loss / step
            print(f"[TrainAE] Epoch {epoch} Step {step}/{len(loader)}  loss={avg:.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device, epoch, phase="ValAE"):
    model.eval()
    total_loss = 0.0

    for numeric, cat, mask, y in loader:
        numeric = numeric.to(device)
        cat = cat.to(device)
        mask = mask.to(device)

        recon = model(numeric, cat, mask)
        loss = masked_mse_loss(recon, numeric, mask)
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"[{phase}] Epoch {epoch}  loss={avg_loss:.4f}")
    return avg_loss


def main():
    # __file__ = D:\NIDS\src\AutoEncoder\train.py
    # parents[0] = AutoEncoder, [1] = src, [2] = NIDS
    ROOT_DIR = Path(__file__).resolve().parents[2]
    PROCESSED_DIR = ROOT_DIR / "processed"

    train_parquet = PROCESSED_DIR / "flows_train_benign.parquet"
    val_parquet   = PROCESSED_DIR / "flows_val_benign.parquet"
    config_path   = PROCESSED_DIR / "preprocess_config.json"

    print("ROOT_DIR      :", ROOT_DIR)
    print("TRAIN_PARQUET :", train_parquet)
    print("VAL_PARQUET   :", val_parquet)
    print("CONFIG_PATH   :", config_path)

    # ------------------------------
    # config 로드
    # ------------------------------
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    numeric_cols = cfg["numeric_cols"]
    numeric_dim_base = len(numeric_cols)   # 원래 flow-level feature 77개
    WINDOW_FEAT_DIM = 5                   # unique_src_count, avg_flows_per_src, src_entropy, window_duration, flow_rate

    numeric_dim = numeric_dim_base + WINDOW_FEAT_DIM  # 77 + 5 = 82

    num_port_classes = cfg["other_port_idx"] + 1
    num_proto_classes = cfg["proto_other_idx"] + 1

    print("\n[Config]")
    print("numeric_dim_base :", numeric_dim_base)
    print("numeric_dim      :", numeric_dim)
    print("num_port_classes :", num_port_classes)
    print("num_proto_classes:", num_proto_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ------------------------------
    # Dataset / DataLoader
    # ------------------------------
    AE_SEQ_LEN = 128
    AE_STRIDE  = 64

    train_ds = FlowWindowDataset(
        parquet_path=str(train_parquet),
        config_path=str(config_path),
        seq_len=AE_SEQ_LEN,
        stride=AE_STRIDE,
    )

    val_ds = FlowWindowDataset(
        parquet_path=str(val_parquet),
        config_path=str(config_path),
        seq_len=AE_SEQ_LEN,
        stride=AE_STRIDE,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ------------------------------
    # Model
    # ------------------------------
    model = TCNAutoencoder(
        numeric_dim=numeric_dim,              # 🔥 82차원으로 변경
        num_port_classes=num_port_classes,
        num_proto_classes=num_proto_classes,
        d_model=128,
        tcn_kernel_size=3,
        tcn_num_layers=3,
        dropout=0.1,
    ).to(device)

    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    num_epochs = 10
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        print(f"\n===== [AE] Epoch {epoch}/{num_epochs} =====")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = validate(model, val_loader, device, epoch, phase="ValAE")

        print(
            f"[SummaryAE] Epoch {epoch} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"best_val_loss={best_val_loss if best_val_loss < float('inf') else float('nan'):.4f}"
        )

        models_dir = ROOT_DIR / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = models_dir / f"ae_tcn_epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[Save] {ckpt_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = models_dir / "ae_tcn_best.pt"
            torch.save(model.state_dict(), best_path)
            print(f"[BestAE Updated] {best_path}  (val_loss={val_loss:.4f})")


if __name__ == "__main__":
    main()
