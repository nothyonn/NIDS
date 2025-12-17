# src/AutoEncoder/test.py

from pathlib import Path
import json
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from preprocess import FlowWindowDataset
from AutoEncoder.model import TCNAutoencoder, masked_mse_loss

# 여러 aggregation 모드를 한 번에 테스트
AGG_MODES = ["mean", "max", "topk"]
TOPK = [3, 5]   # topk일 때만 사용


@torch.no_grad()
def evaluate_ae(model, loader, device, label_classes, agg_mode="mean", topk=3):
    """
    agg_mode: "mean", "max", "topk"
    topk    : agg_mode == "topk"일 때 flow 상위 k개 평균
    """
    model.eval()

    benign_scores = []
    attack_scores = []
    all_scores = []          # 전체 윈도우 점수 (이진 평가용)
    all_labels_binary = []   # 0=benign, 1=attack
    window_scores = []       # 윈도우별 score
    window_labels = []       # 윈도우별 전체 라벨 벡터 (멀티라벨)

    # BENIGN 인덱스
    try:
        benign_idx = label_classes.index("BENIGN")
    except ValueError:
        raise RuntimeError("label_classes 에 'BENIGN' 라벨이 없습니다.")

    for numeric, cat, mask, y in loader:
        numeric = numeric.to(device)
        cat = cat.to(device)
        mask = mask.to(device)
        y = y.to(device)  # [B, C]

        recon = model(numeric, cat, mask)

        # --------------------------------------------------
        # window 단위 reconstruction error 계산 (agg_mode에 따라 다르게)
        # --------------------------------------------------
        mask_exp = mask.unsqueeze(-1)                # [B, L, 1]
        diff2 = ((recon - numeric) ** 2) * mask_exp  # [B, L, F]

        # flow 단위 error: feature dimension 합 -> [B, L]
        flow_err = diff2.sum(dim=2)

        if agg_mode == "mean":
            # 모든 유효 element 평균
            valid_counts = mask_exp.sum(dim=(1, 2)) + 1e-8  # [B]
            per_window_err = diff2.sum(dim=(1, 2)) / valid_counts  # [B]

        elif agg_mode == "max":
            # flow_err 중에서 mask=1인 위치의 최대값 사용
            per_window_err, _ = (flow_err * mask).max(dim=1)   # [B]

        elif agg_mode == "topk":
            # flow_err에서 mask=1 인 위치만 모아서 상위 topk 개 평균
            B, L = flow_err.shape
            per_window_list = []
            for b in range(B):
                valid = mask[b] > 0           # [L] bool
                fe = flow_err[b][valid]       # [M]
                if fe.numel() == 0:
                    per_window_list.append(torch.tensor(0.0, device=device))
                else:
                    k_eff = min(topk, fe.numel())
                    topk_vals, _ = fe.topk(k_eff)
                    per_window_list.append(topk_vals.mean())
            per_window_err = torch.stack(per_window_list, dim=0)  # [B]

        else:
            raise ValueError(f"Unknown agg_mode: {agg_mode}")

        # --------------------------------------------------
        # window의 BENIGN/ATTACK 라벨 결정
        # y: 멀티라벨 → "BENIGN-only" vs "공격 포함"
        # --------------------------------------------------
        label_sum = y.sum(dim=1)  # [B]
        is_benign_only = (label_sum == 1) & (y[:, benign_idx] == 1)
        is_attack = ~is_benign_only

        # numpy 변환
        per_window_err_np = per_window_err.detach().cpu().numpy()  # [B]
        is_benign_only_np = is_benign_only.cpu().numpy()           # [B]
        is_attack_np = is_attack.cpu().numpy()                     # [B]
        y_np = y.cpu().numpy()                                     # [B, C]

        for i in range(len(per_window_err_np)):
            score = per_window_err_np[i]
            b_mask = is_benign_only_np[i]
            a_mask = is_attack_np[i]
            y_vec = y_np[i]

            window_scores.append(score)
            window_labels.append(y_vec)

            if b_mask:
                benign_scores.append(score)
                all_scores.append(score)
                all_labels_binary.append(0)
            elif a_mask:
                attack_scores.append(score)
                all_scores.append(score)
                all_labels_binary.append(1)
            else:
                # (라벨이 애매한 경우 → 스킵)
                continue

    benign_scores = np.array(benign_scores, dtype=np.float32)
    attack_scores = np.array(attack_scores, dtype=np.float32)
    all_scores = np.array(all_scores, dtype=np.float32)
    all_labels_binary = np.array(all_labels_binary, dtype=np.int32)
    window_scores = np.array(window_scores, dtype=np.float32)
    window_labels = np.array(window_labels, dtype=np.int32)  # [N, C]

    print(f"\n===== [AE Anomaly Test Result | agg_mode={agg_mode}] =====")
    print(f"#benign windows : {len(benign_scores)}")
    print(f"#attack windows : {len(attack_scores)}")

    # 기본 통계들
    benign_mean = benign_std = benign_p95 = benign_p99 = None
    attack_mean = attack_std = None
    auc = None

    if len(benign_scores) > 0:
        benign_mean = float(benign_scores.mean())
        benign_std = float(benign_scores.std())
        benign_p95 = float(np.percentile(benign_scores, 95))
        benign_p99 = float(np.percentile(benign_scores, 99))
        print(f"avg benign recon_error : {benign_mean:.6f}")
        print(f"std benign recon_error : {benign_std:.6f}")
        print(f"p95 benign recon_error : {benign_p95:.6f}")
        print(f"p99 benign recon_error : {benign_p99:.6f}")

    if len(attack_scores) > 0:
        attack_mean = float(attack_scores.mean())
        attack_std = float(attack_scores.std())
        print(f"avg attack  recon_error : {attack_mean:.6f}")
        print(f"std attack  recon_error : {attack_std:.6f}")

    if len(np.unique(all_labels_binary)) == 2:
        auc = float(roc_auc_score(all_labels_binary, all_scores))
        print(f"ROC-AUC (benign vs attack) : {auc:.4f}")
    else:
        print("ROC-AUC 계산 불가 (한쪽 클래스만 존재).")

    metrics = {
        "benign_mean": benign_mean,
        "benign_std": benign_std,
        "benign_p95": benign_p95,
        "benign_p99": benign_p99,
        "attack_mean": attack_mean,
        "attack_std": attack_std,
        "roc_auc": auc,
        "benign_idx": benign_idx,
    }

    # metrics + per-window 정보 + 전체 이진 라벨/스코어 반환
    return metrics, window_scores, window_labels, all_scores, all_labels_binary


def print_global_metrics(all_scores, all_labels_binary, threshold):
    """
    전체 윈도우 기준 BENIGN(0) vs ATTACK(1) 이진 분류 성능
    → 출력 + dict 반환 (CSV 저장용)
    """
    scores = np.asarray(all_scores)
    labels = np.asarray(all_labels_binary)  # 0/1
    preds = (scores > threshold).astype(int)

    TP = int(((preds == 1) & (labels == 1)).sum())
    FP = int(((preds == 1) & (labels == 0)).sum())
    TN = int(((preds == 0) & (labels == 0)).sum())
    FN = int(((preds == 0) & (labels == 1)).sum())

    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    print(f"[GLOBAL] TP={TP} FP={FP} TN={TN} FN={FN}")
    print(f"[GLOBAL] precision={prec:.4f}  recall={rec:.4f}  F1={f1:.4f}")

    return {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def analyze_per_class(window_scores, window_labels, label_classes, benign_idx, threshold):
    """
    window_scores: [N]  (recon_error)
    window_labels: [N, C]  (멀티라벨 원-핫)
    threshold: score > T 이면 '이상(attack)'으로 판정

    → 출력 + 클래스별 통계 리스트 반환 (CSV 저장용)
    """
    scores = np.asarray(window_scores)
    labels = np.asarray(window_labels)  # [N, C]
    pred_anom = scores > threshold      # [N] bool

    print(f"\n----- [Per-Class Detection @ T={threshold:.4f}] -----")

    rows = []

    # BENIGN-only 오탐
    label_sum = labels.sum(axis=1)
    is_benign_only = (label_sum == 1) & (labels[:, benign_idx] == 1)
    total_benign_only = int(is_benign_only.sum())
    if total_benign_only > 0:
        fp_benign = int((pred_anom & is_benign_only).sum())
        fp_rate = fp_benign / total_benign_only
        print(
            f"[BENIGN] total={total_benign_only} | "
            f"false_positive={fp_benign} | fp_rate={fp_rate:.4f}"
        )
        rows.append({
            "class": "BENIGN_ONLY",
            "total": total_benign_only,
            "detected": fp_benign,             # '이상으로 판정된 BENIGN'
            "missed": total_benign_only - fp_benign,
            "recall": None,                    # 의미 없음
            "fp_rate": fp_rate,
            "is_benign": True,
        })
    else:
        print("[BENIGN] 샘플 없음")

    # 각 공격 클래스별 탐지율 / 미탐율
    for idx, name in enumerate(label_classes):
        if idx == benign_idx:
            continue  # BENIGN은 위에서 처리

        mask_c = labels[:, idx] == 1
        total_c = int(mask_c.sum())
        if total_c == 0:
            print(f"[{name}] 샘플 없음")
            continue

        detected_c = int((pred_anom & mask_c).sum())
        missed_c = total_c - detected_c
        recall_c = detected_c / total_c

        print(
            f"[{name}] total={total_c} | "
            f"detected={detected_c} | missed={missed_c} | "
            f"recall={recall_c:.4f}"
        )

        rows.append({
            "class": name,
            "total": total_c,
            "detected": detected_c,
            "missed": missed_c,
            "recall": recall_c,
            "fp_rate": None,
            "is_benign": False,
        })

    return rows


def analyze_thresholds(
    window_scores,
    window_labels,
    label_classes,
    benign_idx,
    thresholds,
    all_scores,
    all_labels_binary,
):
    """
    여러 개의 threshold에 대해:
      - Global BENIGN vs ATTACK Precision/Recall/F1
      - Per-class detection 통계
    출력 + CSV 저장용 리스트 반환
    """
    scores = np.asarray(window_scores)
    labels = np.asarray(window_labels)

    global_rows = []
    per_class_rows_all = []

    for T in thresholds:
        print("\n" + "=" * 50)
        print(f"===== [Threshold = {T:.4f}] =====")

        # 1) 전체 이진 분류 기준 글로벌 F1
        global_stats = print_global_metrics(all_scores, all_labels_binary, T)
        global_stats["threshold"] = float(T)
        global_rows.append(global_stats)

        # 2) 클래스별 detection 통계
        per_class_rows = analyze_per_class(scores, labels, label_classes, benign_idx, T)
        for r in per_class_rows:
            r["threshold"] = float(T)
        per_class_rows_all.extend(per_class_rows)

    return global_rows, per_class_rows_all


def main():
    ROOT_DIR = Path(__file__).resolve().parents[2]
    PROCESSED_DIR = ROOT_DIR / "processed"
    RESULTS_DIR = ROOT_DIR / "results" / "ae"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    test_parquet = PROCESSED_DIR / "flows_test_scaled.parquet"
    config_path  = PROCESSED_DIR / "preprocess_config.json"
    best_model_path = ROOT_DIR / "models" / "ae_tcn_best.pt"

    print("ROOT_DIR        :", ROOT_DIR)
    print("TEST_PARQUET    :", test_parquet)
    print("CONFIG_PATH     :", config_path)
    print("BEST_MODEL_PATH :", best_model_path)

    # ------------------------------
    # config 로드
    # ------------------------------
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    numeric_cols = cfg["numeric_cols"]
    numeric_dim_base = len(numeric_cols)
    WINDOW_FEAT_DIM = 5
    numeric_dim = numeric_dim_base + WINDOW_FEAT_DIM

    label_classes = cfg["label_classes"]
    num_port_classes = cfg["other_port_idx"] + 1
    num_proto_classes = cfg["proto_other_idx"] + 1

    print("\n[Config]")
    print("numeric_dim      :", numeric_dim)
    print("num_port_classes :", num_port_classes)
    print("num_proto_classes:", num_proto_classes)
    print("label_classes    :", label_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ------------------------------
    # Dataset / DataLoader
    # ------------------------------
    AE_SEQ_LEN = 128
    AE_STRIDE  = 64

    test_ds = FlowWindowDataset(
        parquet_path=str(test_parquet),
        config_path=str(config_path),
        seq_len=AE_SEQ_LEN,
        stride=AE_STRIDE,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ------------------------------
    # Model 정의 + best weight 로드
    # ------------------------------
    model = TCNAutoencoder(
        numeric_dim=numeric_dim,
        num_port_classes=num_port_classes,
        num_proto_classes=num_proto_classes,
        d_model=128,
        tcn_kernel_size=3,
        tcn_num_layers=3,
        dropout=0.1,
    ).to(device)

    print("\n[Load] Best AE model state_dict...")
    state = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state)

    # ------------------------------
    # agg_mode 및 topk 조합 전체 테스트
    # ------------------------------
    TOPK_LIST = [3, 5]

    for agg_mode in AGG_MODES:

        # ----------- mean / max는 topk 없음 -----------
        if agg_mode != "topk":
            print("\n" + "#" * 80)
            print(f"##################  AGG_MODE = {agg_mode}  ##################")

            metrics, window_scores, window_labels, all_scores, all_labels_binary = evaluate_ae(
                model, test_loader, device, label_classes,
                agg_mode=agg_mode,
                topk=None
            )

            benign_p95 = metrics["benign_p95"]
            benign_p99 = metrics["benign_p99"]
            benign_idx = metrics["benign_idx"]

            thresholds = set()

            benign_mean = metrics["benign_mean"]
            benign_std  = metrics["benign_std"]

            if benign_p95 is not None:
                thresholds.add(benign_p95)
            if benign_p99 is not None:
                thresholds.add(benign_p99)

            if (benign_mean is not None) and (benign_std is not None):
                thresholds.add(benign_mean + 2 * benign_std)
                thresholds.add(benign_mean + 3 * benign_std)

            thresholds.update([0.5, 1.0, 2.0, 3.0, 4.0])
            thresholds = sorted(thresholds)

            print("\n===== [Threshold Sweep | agg_mode="
                  f"{agg_mode}] =====")
            print("thresholds:", ", ".join(f"{t:.4f}" for t in thresholds))

            # 1) 글로벌/클래스별 메트릭 계산 + CSV용 리스트
            global_rows, per_class_rows = analyze_thresholds(
                window_scores, window_labels,
                label_classes, benign_idx,
                thresholds,
                all_scores, all_labels_binary,
            )

            # 2) 점수 분포 CSV 저장 (재구성 오차 vs 이진 라벨)
            base_name = f"agg_{agg_mode}"
            scores_csv = RESULTS_DIR / f"scores_{base_name}.csv"
            with scores_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["score", "label_binary"])
                for s, lb in zip(all_scores, all_labels_binary):
                    writer.writerow([f"{float(s):.6f}", int(lb)])

            # 3) threshold별 글로벌 메트릭 CSV
            global_csv = RESULTS_DIR / f"threshold_global_{base_name}.csv"
            with global_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "threshold", "TP", "FP", "TN", "FN",
                    "precision", "recall", "f1",
                ])
                for row in global_rows:
                    writer.writerow([
                        f"{row['threshold']:.6f}",
                        row["TP"], row["FP"], row["TN"], row["FN"],
                        f"{row['precision']:.6f}",
                        f"{row['recall']:.6f}",
                        f"{row['f1']:.6f}",
                    ])

            # 4) threshold × class별 메트릭 CSV
            per_class_csv = RESULTS_DIR / f"threshold_per_class_{base_name}.csv"
            with per_class_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "threshold", "class", "is_benign",
                    "total", "detected", "missed",
                    "recall", "fp_rate",
                ])
                for r in per_class_rows:
                    writer.writerow([
                        f"{r['threshold']:.6f}",
                        r["class"],
                        int(r["is_benign"]),
                        r["total"],
                        r["detected"],
                        r["missed"],
                        "" if r["recall"] is None else f"{r['recall']:.6f}",
                        "" if r["fp_rate"] is None else f"{r['fp_rate']:.6f}",
                    ])

            if metrics["roc_auc"] is not None:
                print(f"\n(참고) ROC-AUC(agg_mode={agg_mode}) = {metrics['roc_auc']:.4f}")

        # ----------- topk의 경우 topk 리스트 전체 반복 -----------
        else:
            for topk_value in TOPK_LIST:
                print("\n" + "#" * 80)
                print(f"##########  AGG_MODE = topk, TOPK = {topk_value}  ##########")

                metrics, window_scores, window_labels, all_scores, all_labels_binary = evaluate_ae(
                    model, test_loader, device, label_classes,
                    agg_mode="topk",
                    topk=topk_value
                )

                benign_p95 = metrics["benign_p95"]
                benign_p99 = metrics["benign_p99"]
                benign_idx = metrics["benign_idx"]

                thresholds = set()

                benign_mean = metrics["benign_mean"]
                benign_std  = metrics["benign_std"]

                if benign_p95 is not None:
                    thresholds.add(benign_p95)
                if benign_p99 is not None:
                    thresholds.add(benign_p99)

                if (benign_mean is not None) and (benign_std is not None):
                    thresholds.add(benign_mean + 2 * benign_std)
                    thresholds.add(benign_mean + 3 * benign_std)

                thresholds.update([0.5, 1.0, 2.0, 3.0, 4.0])
                thresholds = sorted(thresholds)

                print("\n===== [Threshold Sweep | agg_mode=topk, TOPK="
                      f"{topk_value}] =====")
                print("thresholds:", ", ".join(f"{t:.4f}" for t in thresholds))

                global_rows, per_class_rows = analyze_thresholds(
                    window_scores, window_labels,
                    label_classes, benign_idx,
                    thresholds,
                    all_scores, all_labels_binary,
                )

                base_name = f"agg_topk{topk_value}"
                # 점수 분포 CSV
                scores_csv = RESULTS_DIR / f"scores_{base_name}.csv"
                with scores_csv.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["score", "label_binary"])
                    for s, lb in zip(all_scores, all_labels_binary):
                        writer.writerow([f"{float(s):.6f}", int(lb)])

                # threshold 글로벌 메트릭 CSV
                global_csv = RESULTS_DIR / f"threshold_global_{base_name}.csv"
                with global_csv.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "threshold", "TP", "FP", "TN", "FN",
                        "precision", "recall", "f1",
                    ])
                    for row in global_rows:
                        writer.writerow([
                            f"{row['threshold']:.6f}",
                            row["TP"], row["FP"], row["TN"], row["FN"],
                            f"{row['precision']:.6f}",
                            f"{row['recall']:.6f}",
                            f"{row['f1']:.6f}",
                        ])

                # threshold × class별 메트릭 CSV
                per_class_csv = RESULTS_DIR / f"threshold_per_class_{base_name}.csv"
                with per_class_csv.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "threshold", "class", "is_benign",
                        "total", "detected", "missed",
                        "recall", "fp_rate",
                    ])
                    for r in per_class_rows:
                        writer.writerow([
                            f"{r['threshold']:.6f}",
                            r["class"],
                            int(r["is_benign"]),
                            r["total"],
                            r["detected"],
                            r["missed"],
                            "" if r["recall"] is None else f"{r['recall']:.6f}",
                            "" if r["fp_rate"] is None else f"{r['fp_rate']:.6f}",
                        ])

                if metrics["roc_auc"] is not None:
                    print(f"\n(참고) ROC-AUC(agg_mode=topk, topk={topk_value}) = {metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
