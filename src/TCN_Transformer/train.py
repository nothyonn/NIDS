# src/tcn_transformer/train.py

from pathlib import Path
import json
import numpy as np

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from preprocess import FlowWindowDataset
from TCN_Transformer import TCNTransformerModel


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0

    for step, (numeric, cat, mask, y) in enumerate(loader, 1):
        numeric = numeric.to(device)  # [B, L, F]
        cat = cat.to(device)          # [B, L, 3]
        mask = mask.to(device)        # [B, L]
        y = y.to(device)              # [B, C]

        optimizer.zero_grad()
        logits = model(numeric, cat, mask)  # [B, C]
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 100 == 0:
            avg = total_loss / step
            print(f"[Train] Epoch {epoch} Step {step}/{len(loader)}  loss={avg:.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    """
    - BCE loss
    - subset accuracy  (윈도우의 멀티라벨을 전부 맞춘 비율)
    - micro / macro F1
    - label_hit_ratio: 윈도우별로 '실제 라벨 중 몇 개를 맞췄는지' 평균

      예) 실제 라벨: [BruteForce, DDoS, Hulk] = 3개
          예측 라벨: [DDoS, Hulk]                → 2/3
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    subset_correct = 0

    all_targets = []
    all_preds = []

    # label hit ratio 계산용
    sum_hit_ratio = 0.0
    count_hit_windows = 0

    for numeric, cat, mask, y in loader:
        numeric = numeric.to(device)
        cat = cat.to(device)
        mask = mask.to(device)
        y = y.to(device)

        logits = model(numeric, cat, mask)
        loss = criterion(logits, y)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()        # [B, C]

        # subset accuracy (모든 라벨이 정확히 같을 때만 정답)
        subset_correct += ((preds == y).all(dim=1)).sum().item()
        total_samples += y.size(0)

        # ----- label hit ratio -----
        # 각 윈도우별로: (TP 수 / 실제 양성 라벨 수)
        # 실제 라벨 수가 0인 윈도우(BENIGN-only) 는 계산에서 제외
        with torch.no_grad():
            true_pos_per_sample = ((preds == 1) & (y == 1)).sum(dim=1)      # [B]
            true_labels_per_sample = (y == 1).sum(dim=1)                    # [B]

            mask_has_label = true_labels_per_sample > 0                      # 라벨이 1개 이상 있는 윈도우만 사용
            if mask_has_label.any():
                hit_ratio = (
                    true_pos_per_sample[mask_has_label].float()
                    / true_labels_per_sample[mask_has_label].float()
                )  # [N_hit]
                sum_hit_ratio += hit_ratio.sum().item()
                count_hit_windows += hit_ratio.numel()

        # F1 계산용 버퍼
        all_targets.append(y.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    avg_loss = total_loss / len(loader)
    subset_acc = subset_correct / max(1, total_samples)

    # ---- micro / macro F1 ----
    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)

    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # ---- label_hit_ratio ----
    if count_hit_windows > 0:
        label_hit_ratio = sum_hit_ratio / count_hit_windows
    else:
        label_hit_ratio = 0.0

    print(
        f"[Val]   Epoch {epoch}  "
        f"loss={avg_loss:.4f}  "
        f"subset_acc={subset_acc:.4f}  "
        f"micro_F1={micro_f1:.4f}  "
        f"macro_F1={macro_f1:.4f}  "
        f"label_hit_ratio={label_hit_ratio:.4f}"
    )

    return avg_loss, subset_acc, micro_f1, macro_f1, label_hit_ratio


def main():
    # __file__ = D:\NIDS\src\tcn_transformer\train.py
    # parents[0]=tcn_transformer, [1]=src, [2]=NIDS
    ROOT_DIR = Path(__file__).resolve().parents[2]  # D:\NIDS
    PROCESSED_DIR = ROOT_DIR / "processed"

    # 🔹 오용탐지용 세트(세트1) parquet (make_misuse_sets.py에서 만든 파일)
    train_parquet = PROCESSED_DIR / "flows_train_misuse.parquet"
    val_parquet   = PROCESSED_DIR / "flows_val_misuse.parquet"

    # 🔹 전처리 설정 (numeric cols, port/proto idx, label_classes 등)
    config_path   = PROCESSED_DIR / "preprocess_config.json"

    # ------------------------------
    # config 로드
    # ------------------------------
    with open(config_path, "r") as f:
        cfg = json.load(f)

    numeric_cols = cfg["numeric_cols"]
    label_classes = cfg["label_classes"]

    numeric_dim_base = len(numeric_cols)
    WINDOW_FEAT_DIM = 5

    numeric_dim = numeric_dim_base + WINDOW_FEAT_DIM
    num_classes = len(label_classes)

    # Port / Protocol vocab 크기 (idx는 0 ~ other_idx)
    num_port_classes = cfg["other_port_idx"] + 1
    num_proto_classes = cfg["proto_other_idx"] + 1

    print("\n[Config]")
    print("numeric_dim_base :", numeric_dim_base)
    print("numeric_dim      :", numeric_dim)
    print("num_classes      :", num_classes)
    print("num_port_classes :", num_port_classes)
    print("num_proto_classes:", num_proto_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ------------------------------
    # Dataset / DataLoader
    # ------------------------------
    train_ds = FlowWindowDataset(
        parquet_path=str(train_parquet),
        config_path=str(config_path),
        seq_len=128,
        stride=64,
    )

    val_ds = FlowWindowDataset(
        parquet_path=str(val_parquet),
        config_path=str(config_path),
        seq_len=128,
        stride=64,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=0,   # 윈도우면 0 추천
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    # ------------------------------
    # Model
    # ------------------------------
    model = TCNTransformerModel(
        numeric_dim=numeric_dim,
        num_classes=num_classes,
        num_port_classes=num_port_classes,
        num_proto_classes=num_proto_classes,
        d_model=128,
        num_heads=4,
        num_layers=2,
        tcn_channels=128,
        tcn_kernel_size=3,
        tcn_num_layers=2,
        dropout=0.1,
        max_len=129,  # 128 flows + 1 CLS
    ).to(device)

    print(model)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # ------------------------------
    # Train loop
    # ------------------------------
    num_epochs = 5  # 노트북에서는 5 정도, 데탑 GPU에서 늘리면 됨

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        print(f"\n===== Epoch {epoch}/{num_epochs} =====")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_subset_acc, micro_f1, macro_f1, label_hit_ratio = validate(
            model, val_loader, criterion, device, epoch
        )

        # 요약 로그
        print(
            f"[Summary] Epoch {epoch} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"subset_acc={val_subset_acc:.4f} | "
            f"micro_F1={micro_f1:.4f} | "
            f"macro_F1={macro_f1:.4f} | "
            f"label_hit_ratio={label_hit_ratio:.4f}"
        )

        # 모델 저장
        models_dir = ROOT_DIR / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = models_dir / f"tcn_transformer_v2_epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[Save] {ckpt_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = models_dir / "tcn_transformer_v2_best.pt"
            torch.save(model.state_dict(), best_path)
            print(f"[Best Updated] {best_path}  (val_loss={val_loss:.4f})")


if __name__ == "__main__":
    main()
