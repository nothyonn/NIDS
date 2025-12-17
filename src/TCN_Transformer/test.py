# src/tcn_transformer/test.py

from pathlib import Path
import json
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score

from preprocess import FlowWindowDataset
from TCN_Transformer import TCNTransformerModel


@torch.no_grad()
def evaluate_misuse(model, loader, criterion, device, label_classes):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    subset_correct = 0

    all_targets = []
    all_preds = []

    for numeric, cat, mask, y in loader:
        numeric = numeric.to(device)
        cat = cat.to(device)
        mask = mask.to(device)
        y = y.to(device)

        logits = model(numeric, cat, mask)
        loss = criterion(logits, y)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()      # [B, C]

        # subset accuracy (윈도우 내 모든 라벨 완전 일치 비율)
        subset_correct += ((preds == y).all(dim=1)).sum().item()
        total_samples += y.size(0)

        all_targets.append(y.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    avg_loss = total_loss / max(1, len(loader))
    subset_acc = subset_correct / max(1, total_samples)

    y_true = np.concatenate(all_targets, axis=0)  # [N, C]
    y_pred = np.concatenate(all_preds, axis=0)    # [N, C]

    # ------------------------------
    # 1) micro / macro F1
    # ------------------------------
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # ------------------------------
    # 2) micro / macro recall (전체 라벨 기준)
    # ------------------------------
    micro_recall = recall_score(y_true, y_pred, average="micro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # ------------------------------
    # 3) label_hit_ratio (윈도우 기준)
    # ------------------------------
    tp_per_sample = ((y_true == 1) & (y_pred == 1)).sum(axis=1)
    true_pos_per_sample = (y_true == 1).sum(axis=1)
    valid_mask = true_pos_per_sample > 0
    if valid_mask.any():
        label_hit_ratio = (tp_per_sample[valid_mask] / true_pos_per_sample[valid_mask]).mean()
    else:
        label_hit_ratio = 0.0

    # ------------------------------
    # 4) 클래스별 F1 / Recall / support
    # ------------------------------
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_support = y_true.sum(axis=0).astype(int)

    # ------------------------------
    # 5) BENIGN vs Any Attack 관점 (fusion에서 쓰는 정의랑 맞춤)
    #    - 공격 라벨이 하나라도 있으면 Attack 윈도우
    # ------------------------------
    # label_classes[0] = 'BENIGN' 이라는 전제 사용
    true_attack_any = (y_true[:, 1:] == 1).any(axis=1)   # [N]
    pred_attack_any = (y_pred[:, 1:] == 1).any(axis=1)   # [N]

    total_attack = int(true_attack_any.sum())
    tcn_tp_any = int((true_attack_any & pred_attack_any).sum())
    tcn_fn_any = int((true_attack_any & ~pred_attack_any).sum())

    denom = (tcn_tp_any + tcn_fn_any)
    if denom > 0:
        attack_any_recall = tcn_tp_any / denom
        attack_any_fn_rate = tcn_fn_any / denom
    else:
        attack_any_recall = 0.0
        attack_any_fn_rate = 0.0

    # ------------------------------
    # 로그 출력
    # ------------------------------
    print("\n===== [Misuse Test Result] =====")
    print(f"loss                : {avg_loss:.4f}")
    print(f"subset_acc          : {subset_acc:.4f}")
    print(f"micro_F1            : {micro_f1:.4f}")
    print(f"macro_F1            : {macro_f1:.4f}")
    print(f"micro_recall        : {micro_recall:.4f}")
    print(f"macro_recall        : {macro_recall:.4f}")
    print(f"label_hit_ratio     : {label_hit_ratio:.4f}")
    print(f"attack_any_recall   : {attack_any_recall:.4f}")
    print(f"attack_any_FN_rate  : {attack_any_fn_rate:.4f}")
    print(f"total_attack_windows: {total_attack}")
    print("\n[Per-class F1 / Recall]")
    for name, f1, r, sup in zip(label_classes, per_class_f1, per_class_recall, per_class_support):
        print(f"  {name:30s}: F1={f1:.4f}, recall={r:.4f}, support={sup}")

    return {
        "loss": avg_loss,
        "subset_acc": subset_acc,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "micro_recall": micro_recall,
        "macro_recall": macro_recall,
        "label_hit_ratio": label_hit_ratio,
        "attack_any_recall": attack_any_recall,
        "attack_any_fn_rate": attack_any_fn_rate,
        "total_attack": total_attack,
        "per_class_f1": dict(zip(label_classes, per_class_f1)),
        "per_class_recall": dict(zip(label_classes, per_class_recall)),
        "per_class_support": dict(zip(label_classes, per_class_support)),
    }


def main():
    # 현재 파일: D:/NIDS/src/tcn_transformer/test.py
    # parents[0] = tcn_transformer, parents[1] = src, parents[2] = NIDS
    ROOT_DIR = Path(__file__).resolve().parents[2]
    PROCESSED_DIR = ROOT_DIR / "processed"
    RESULTS_DIR = ROOT_DIR / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    test_parquet = PROCESSED_DIR / "flows_test_misuse.parquet"
    config_path  = PROCESSED_DIR / "preprocess_config.json"
    best_model_path = ROOT_DIR / "models" / "tcn_transformer_v2_best.pt"

    print("ROOT_DIR        :", ROOT_DIR)
    print("TEST_PARQUET    :", test_parquet)
    print("CONFIG_PATH     :", config_path)
    print("BEST_MODEL_PATH :", best_model_path)
    print("RESULTS_DIR     :", RESULTS_DIR)

    # ------------------------------
    # config 로드
    # ------------------------------
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    numeric_cols = cfg["numeric_cols"]
    label_classes = cfg["label_classes"]

    numeric_dim_base = len(numeric_cols)
    WINDOW_FEAT_DIM = 5

    numeric_dim = numeric_dim_base + WINDOW_FEAT_DIM
    num_classes = len(label_classes)

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
    # Dataset / DataLoader (misuse test)
    # ------------------------------
    test_ds = FlowWindowDataset(
        parquet_path=str(test_parquet),
        config_path=str(config_path),
        seq_len=128,
        stride=64,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    # ------------------------------
    # Model 정의 + best weight 로드
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

    print("\n[Load] Best model state_dict...")
    state = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state)

    criterion = torch.nn.BCEWithLogitsLoss()

    # ------------------------------
    # Misuse Test 평가
    # ------------------------------
    metrics = evaluate_misuse(model, test_loader, criterion, device, label_classes)

    # ------------------------------
    # CSV 저장
    # ------------------------------
    summary_path = RESULTS_DIR / "tcn_misuse_summary.csv"
    per_class_path = RESULTS_DIR / "tcn_misuse_per_class.csv"

    # 1) 요약 메트릭 (single row)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "loss",
            "subset_acc",
            "micro_f1",
            "macro_f1",
            "micro_recall",
            "macro_recall",
            "label_hit_ratio",
            "attack_any_recall",
            "attack_any_fn_rate",
            "total_attack",
        ])
        writer.writerow([
            metrics["loss"],
            metrics["subset_acc"],
            metrics["micro_f1"],
            metrics["macro_f1"],
            metrics["micro_recall"],
            metrics["macro_recall"],
            metrics["label_hit_ratio"],
            metrics["attack_any_recall"],
            metrics["attack_any_fn_rate"],
            metrics["total_attack"],
        ])

    # 2) 클래스별 메트릭
    with per_class_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "f1", "recall", "support"])
        for cls in label_classes:
            writer.writerow([
                cls,
                metrics["per_class_f1"][cls],
                metrics["per_class_recall"][cls],
                metrics["per_class_support"][cls],
            ])

    print(f"\n[CSV Saved] {summary_path}")
    print(f"[CSV Saved] {per_class_path}")


if __name__ == "__main__":
    main()
