# src/engine/fusion_test.py

from pathlib import Path
import json
import numpy as np
import csv

import torch
from torch.utils.data import DataLoader

from preprocess import FlowWindowDataset
from TCN_Transformer import TCNTransformerModel
from AutoEncoder import TCNAutoencoder

from engine.fusion_model import compute_ae_scores, update_recall_stats


@torch.no_grad()
def run_fusion_test():
    # __file__ = D:\NIDS\src\engine\fusion_test.py
    ROOT_DIR = Path(__file__).resolve().parents[2]  # D:\NIDS
    PROCESSED_DIR = ROOT_DIR / "processed"
    MODELS_DIR = ROOT_DIR / "models"

    # [NEW] 결과 csv 저장할 폴더
    RESULTS_DIR = ROOT_DIR / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    TCN_PARQUET = PROCESSED_DIR / "flows_test_scaled.parquet"
    AE_PARQUET = PROCESSED_DIR / "flows_test_scaled.parquet"
    CONFIG_PATH = PROCESSED_DIR / "preprocess_config.json"
    TCN_MODEL_PATH = MODELS_DIR / "tcn_transformer_v2_best.pt"
    AE_MODEL_PATH = MODELS_DIR / "ae_tcn_best.pt"

    print("ROOT_DIR        :", ROOT_DIR)
    print("TCN_PARQUET     :", TCN_PARQUET)
    print("AE_PARQUET      :", AE_PARQUET)
    print("CONFIG_PATH     :", CONFIG_PATH)
    print("TCN_MODEL_PATH  :", TCN_MODEL_PATH)
    print("AE_MODEL_PATH   :", AE_MODEL_PATH)
    print()

    # ------------------------------
    # config 로드
    # ------------------------------
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    numeric_cols = cfg["numeric_cols"]
    label_classes = cfg["label_classes"]

    numeric_dim_base = len(numeric_cols)
    WINDOW_FEAT_DIM = 5
    numeric_dim = numeric_dim_base + WINDOW_FEAT_DIM
    num_classes = len(label_classes)

    num_port_classes = cfg["other_port_idx"] + 1
    num_proto_classes = cfg["proto_other_idx"] + 1

    print("[Config]")
    print("numeric_dim_base :", numeric_dim_base)
    print("numeric_dim      :", numeric_dim)
    print("num_classes      :", num_classes)
    print("num_port_classes :", num_port_classes)
    print("num_proto_classes:", num_proto_classes)
    print("label_classes    :", label_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ------------------------------
    # Dataset / DataLoader
    # ------------------------------
    SEQ_LEN = 128
    STRIDE = 64

    ds_tcn = FlowWindowDataset(
        parquet_path=str(TCN_PARQUET),
        config_path=str(CONFIG_PATH),
        seq_len=SEQ_LEN,
        stride=STRIDE,
    )
    ds_ae = FlowWindowDataset(
        parquet_path=str(AE_PARQUET),
        config_path=str(CONFIG_PATH),
        seq_len=SEQ_LEN,
        stride=STRIDE,
    )

    print(f"[FlowWindowDataset] TCN windows : {len(ds_tcn)}")
    print(f"[FlowWindowDataset] AE  windows : {len(ds_ae)}")

    assert len(ds_tcn) == len(ds_ae), "TCN / AE dataset 길이가 다릅니다."

    loader_tcn = DataLoader(
        ds_tcn,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    loader_ae = DataLoader(
        ds_ae,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ------------------------------
    # 모델 로드
    # ------------------------------
    tcn_model = TCNTransformerModel(
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
        max_len=SEQ_LEN + 1,
    ).to(device)

    print("\n[Load] Best TCN-Transformer model state_dict...")
    state_tcn = torch.load(TCN_MODEL_PATH, map_location=device)
    tcn_model.load_state_dict(state_tcn)
    tcn_model.eval()

    ae_model = TCNAutoencoder(
        numeric_dim=numeric_dim,
        num_port_classes=num_port_classes,
        num_proto_classes=num_proto_classes,
        d_model=128,
        tcn_kernel_size=3,
        tcn_num_layers=3,
        dropout=0.1,
    ).to(device)

    print("[Load] Best AE model state_dict...")
    state_ae = torch.load(AE_MODEL_PATH, map_location=device)
    ae_model.load_state_dict(state_ae)
    ae_model.eval()

    # ------------------------------
    # Fusion 설정 (AE 테스트 때와 동일하게)
    # ------------------------------
    agg_mode = "topk"
    topk = 3
    threshold = 1.5624  # BENIGN p95 (AE 단독 테스트에서 사용한 값)

    print()
    print(
        f"[AE Fusion Config] agg_mode={agg_mode}, topk={topk}, threshold={threshold:.4f}"
    )

    # ------------------------------
    # 통계 집계 변수
    # ------------------------------
    label_classes_arr = np.array(label_classes)
    attack_indices = list(range(1, num_classes))  # 1~C-1 (BENIGN 제외)

    total_attack_all = 0
    tcn_tp_all = 0
    tcn_fn_all = 0

    per_class_total_all = np.zeros(len(attack_indices), dtype=np.int64)
    per_class_tcn_tp_all = np.zeros(len(attack_indices), dtype=np.int64)
    per_class_ae_recover_all = np.zeros(len(attack_indices), dtype=np.int64)

    # --- BENIGN vs ATTACK 이진 confusion (TCN / Fusion) ---
    tcn_TP_any = tcn_FP_any = tcn_TN_any = tcn_FN_any = 0
    fusion_TP_any = fusion_FP_any = fusion_TN_any = fusion_FN_any = 0

    # ------------------------------
    # 메인 루프
    # ------------------------------
    for (batch_tcn, batch_ae) in zip(loader_tcn, loader_ae):
        numeric_tcn, cat_tcn, mask_tcn, y = batch_tcn
        numeric_ae, cat_ae, mask_ae, y2 = batch_ae

        # sanity check: 라벨 동일한지
        assert torch.equal(y, y2), "TCN / AE 라벨 불일치 발생"

        numeric_tcn = numeric_tcn.to(device)
        cat_tcn = cat_tcn.to(device)
        mask_tcn = mask_tcn.to(device)
        y = y.to(device)

        numeric_ae = numeric_ae.to(device)
        cat_ae = cat_ae.to(device)
        mask_ae = mask_ae.to(device)

        # -------- TCN 추론 --------
        logits_tcn = tcn_model(numeric_tcn, cat_tcn, mask_tcn)  # [B, C]
        probs_tcn = torch.sigmoid(logits_tcn)
        preds_tcn = (probs_tcn >= 0.5).float()  # [B, C]

        # 공격 윈도우 마스크 (GT 기준)
        true_attack_any = (y[:, 1:] == 1).any(dim=1)  # [B] bool

        # TCN이 공격이라고 예측한 윈도우
        pred_attack_any = (preds_tcn[:, 1:] == 1).any(dim=1)  # [B] bool

        # TCN 기준 TP/FN mask
        tcn_tp_mask = true_attack_any & pred_attack_any
        tcn_fn_mask = true_attack_any & (~pred_attack_any)

        # -------- AE: TCN이 놓친 윈도우들만 --------
        idx_missed = torch.nonzero(tcn_fn_mask, as_tuple=True)[0]  # [N_missed]
        if idx_missed.numel() > 0:
            recon = ae_model(
                numeric_ae[idx_missed],
                cat_ae[idx_missed],
                mask_ae[idx_missed],
            )  # [N_missed, L, F]

            scores = compute_ae_scores(
                recon=recon,
                target=numeric_ae[idx_missed],
                mask=mask_ae[idx_missed],
                agg_mode=agg_mode,
                topk=topk,
            )  # [N_missed]

            ae_anomaly_mask = scores >= threshold  # [N_missed] bool

            ae_recovered_mask_batch = torch.zeros_like(
                true_attack_any, dtype=torch.bool
            )
            ae_recovered_mask_batch[idx_missed] = ae_anomaly_mask
        else:
            ae_recovered_mask_batch = torch.zeros_like(
                true_attack_any, dtype=torch.bool
            )

        # -------- Fusion 기준 이진 예측 (BENIGN vs ATTACK) --------
        fusion_pred_attack_any = pred_attack_any | ae_recovered_mask_batch  # [B] bool

        # -------- TCN / Fusion confusion matrix 집계 --------
        # TCN
        tcn_TP_any += int((true_attack_any & pred_attack_any).sum().item())
        tcn_FP_any += int((~true_attack_any & pred_attack_any).sum().item())
        tcn_TN_any += int((~true_attack_any & ~pred_attack_any).sum().item())
        tcn_FN_any += int((true_attack_any & ~pred_attack_any).sum().item())

        # Fusion
        fusion_TP_any += int(
            (true_attack_any & fusion_pred_attack_any).sum().item()
        )
        fusion_FP_any += int(
            (~true_attack_any & fusion_pred_attack_any).sum().item()
        )
        fusion_TN_any += int(
            (~true_attack_any & ~fusion_pred_attack_any).sum().item()
        )
        fusion_FN_any += int(
            (true_attack_any & ~fusion_pred_attack_any).sum().item()
        )

        # -------- 통계 집계 (per-class recall 등) --------
        (
            total_attack,
            tcn_tp,
            tcn_fn,
            per_class_total,
            per_class_tcn_tp,
            per_class_ae_recover,
        ) = update_recall_stats(y, preds_tcn, ae_recovered_mask_batch)

        total_attack_all += total_attack
        tcn_tp_all += tcn_tp
        tcn_fn_all += tcn_fn
        per_class_total_all += per_class_total
        per_class_tcn_tp_all += per_class_tcn_tp
        per_class_ae_recover_all += per_class_ae_recover

    # ------------------------------
    # 최종 통계 출력 (Fusion recall, per-class recall + recover)
    # ------------------------------
    if total_attack_all == 0:
        print("공격 윈도우가 하나도 없습니다. (total_attack_all == 0)")
        return

    base_recall = tcn_tp_all / total_attack_all
    fusion_recall_any = (tcn_tp_all + per_class_ae_recover_all.sum()) / total_attack_all

    print()
    print("==================== [Fusion Result Summary] ====================")
    print(f"총 공격 윈도우 수          : {total_attack_all}")
    print(f"TCN 단독 TP               : {tcn_tp_all}")
    print(f"TCN 단독 FN               : {tcn_fn_all}")
    print(f"TCN이 놓친 것 중 AE가 잡은 수 : {int(per_class_ae_recover_all.sum())}")
    print(f"기본 TCN recall (any attack)      : {base_recall:.4f}")
    print(f"TCN+AE fusion recall (any attack) : {fusion_recall_any:.4f}")
    print()

    # -------- (2) Per-Class Fusion Recall + Recover Ratio --------
    print("-------- [Per-Class Fusion Recall + Recover Ratio] --------")

    # [NEW] per-class csv 준비
    per_class_csv = RESULTS_DIR / "fusion_per_class.csv"
    with open(per_class_csv, "w", newline="", encoding="utf-8") as f_pc:
        w_pc = csv.writer(f_pc)
        w_pc.writerow([
            "class",
            "total_attack_windows",
            "tcn_tp",
            "tcn_fn",
            "ae_recovered",
            "base_recall",
            "fusion_recall",
            "recover_ratio",      # AE가 TCN-missed 중 몇 %를 회복했는지 (0~1)
            "recover_ratio_pct",  # 위를 %로
        ])

        for local_idx, c in enumerate(attack_indices):
            cls_name = label_classes_arr[c]
            total_c = int(per_class_total_all[local_idx])
            tcn_tp_c = int(per_class_tcn_tp_all[local_idx])
            ae_rec_c = int(per_class_ae_recover_all[local_idx])

            if total_c == 0:
                base_r = 0.0
                fusion_r = 0.0
                recover_ratio = 0.0
            else:
                base_r = tcn_tp_c / total_c
                fusion_r = (tcn_tp_c + ae_rec_c) / total_c
                fn_c = total_c - tcn_tp_c
                recover_ratio = (ae_rec_c / fn_c) if fn_c > 0 else 0.0

            print(
                f"[{cls_name:<24}] "
                f"fusion_recall={fusion_r:.4f}  "
                f"(recover_ratio={recover_ratio*100:5.1f}% of TCN-missed, "
                f"base_recall={base_r:.4f})"
            )

            # csv 한 줄 기록
            w_pc.writerow([
                cls_name,
                total_c,
                tcn_tp_c,
                total_c - tcn_tp_c,        # tcn_fn
                ae_rec_c,
                f"{base_r:.6f}",
                f"{fusion_r:.6f}",
                f"{recover_ratio:.6f}",
                f"{recover_ratio*100:.2f}",
            ])

    # -------- (1) Hybrid 전체 Precision / Recall / F1 (BENIGN vs ATTACK) --------
    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    # TCN-only (baseline)
    tcn_prec_any = safe_div(tcn_TP_any, tcn_TP_any + tcn_FP_any)
    tcn_rec_any = safe_div(tcn_TP_any, tcn_TP_any + tcn_FN_any)
    tcn_f1_any = safe_div(
        2 * tcn_prec_any * tcn_rec_any, tcn_prec_any + tcn_rec_any
    )

    # Fusion (Hybrid)
    fusion_prec_any = safe_div(fusion_TP_any, fusion_TP_any + fusion_FP_any)
    fusion_rec_any = safe_div(fusion_TP_any, fusion_TP_any + fusion_FN_any)
    fusion_f1_any = safe_div(
        2 * fusion_prec_any * fusion_rec_any,
        fusion_prec_any + fusion_rec_any,
    )

    print()
    print("===== [Hybrid (TCN+AE) – BENIGN vs Any Attack] =====")
    print(f"(Baseline TCN) TP={tcn_TP_any} FP={tcn_FP_any} "
          f"TN={tcn_TN_any} FN={tcn_FN_any}")
    print(
        f"(Baseline TCN) precision={tcn_prec_any:.4f}  "
        f"recall={tcn_rec_any:.4f}  F1={tcn_f1_any:.4f}"
    )

    print(f"\n(Hybrid TCN+AE) TP={fusion_TP_any} FP={fusion_FP_any} "
          f"TN={fusion_TN_any} FN={fusion_FN_any}")
    print(
        f"(Hybrid TCN+AE) precision={fusion_prec_any:.4f}  "
        f"recall={fusion_rec_any:.4f}  F1={fusion_f1_any:.4f}"
    )

    print(
        f"\n[Delta vs TCN]  ΔTP={fusion_TP_any - tcn_TP_any} "
        f"ΔFP={fusion_FP_any - tcn_FP_any} "
        f"ΔFN={fusion_FN_any - tcn_FN_any}"
    )

    # [NEW] 전체 요약 csv
    summary_csv = RESULTS_DIR / "fusion_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f_sum:
        w_sum = csv.writer(f_sum)
        w_sum.writerow([
            "agg_mode",
            "topk",
            "threshold",
            "total_attack_windows",
            "tcn_TP_any",
            "tcn_FP_any",
            "tcn_TN_any",
            "tcn_FN_any",
            "tcn_precision",
            "tcn_recall",
            "tcn_f1",
            "fusion_TP_any",
            "fusion_FP_any",
            "fusion_TN_any",
            "fusion_FN_any",
            "fusion_precision",
            "fusion_recall",
            "fusion_f1",
            "base_any_recall",
            "fusion_any_recall",
        ])
        w_sum.writerow([
            agg_mode,
            topk,
            f"{threshold:.6f}",
            int(total_attack_all),
            int(tcn_TP_any),
            int(tcn_FP_any),
            int(tcn_TN_any),
            int(tcn_FN_any),
            f"{tcn_prec_any:.6f}",
            f"{tcn_rec_any:.6f}",
            f"{tcn_f1_any:.6f}",
            int(fusion_TP_any),
            int(fusion_FP_any),
            int(fusion_TN_any),
            int(fusion_FN_any),
            f"{fusion_prec_any:.6f}",
            f"{fusion_rec_any:.6f}",
            f"{fusion_f1_any:.6f}",
            f"{base_recall:.6f}",
            f"{fusion_recall_any:.6f}",
        ])

    # =====================================================
    # [옵션] AE-only Result (same scores / threshold)
    # =====================================================
    DEBUG_AE_ONLY = False  # 필요하면 True로 바꿔서 참고
    if DEBUG_AE_ONLY:
        print()
        print("============= [DEBUG] AE-only Result (same scores / threshold) =============")

        ae_total_attack = 0
        ae_tp_total = 0

        ae_per_class_total = np.zeros(len(attack_indices), dtype=np.int64)
        ae_per_class_tp = np.zeros(len(attack_indices), dtype=np.int64)

        for batch_ae in loader_ae:
            numeric_ae, cat_ae, mask_ae, y = batch_ae

            numeric_ae = numeric_ae.to(device)
            cat_ae = cat_ae.to(device)
            mask_ae = mask_ae.to(device)
            y = y.to(device)

            recon = ae_model(numeric_ae, cat_ae, mask_ae)  # [B, L, F]

            scores = compute_ae_scores(
                recon=recon,
                target=numeric_ae,
                mask=mask_ae,
                agg_mode=agg_mode,
                topk=topk,
            )  # [B]

            ae_anomaly = (scores >= threshold)  # [B]

            y_np = y.cpu().numpy()
            ae_anom_np = ae_anomaly.cpu().numpy()

            true_attack_any = (y_np[:, 1:] == 1).any(axis=1)  # [B]

            ae_total_attack += int(true_attack_any.sum())
            ae_tp_total += int((true_attack_any & ae_anom_np).sum())

            for local_idx, c in enumerate(attack_indices):
                true_c = (y_np[:, c] == 1)
                ae_per_class_total[local_idx] += int(true_c.sum())
                ae_per_class_tp[local_idx] += int((true_c & ae_anom_np).sum())

        if ae_total_attack == 0:
            print("[AE-only] 공격 윈도우가 없습니다.")
        else:
            ae_recall_any = ae_tp_total / ae_total_attack
            print(f"[AE-only] 총 공격 윈도우 수 : {ae_total_attack}")
            print(f"[AE-only] AE 단독 TP        : {ae_tp_total}")
            print(f"[AE-only] AE 단독 recall   : {ae_recall_any:.4f}")
            print()

        print("----------- [AE-only Per-Class Recall] -----------")
        for local_idx, c in enumerate(attack_indices):
            cls_name = label_classes_arr[c]
            total_c = ae_per_class_total[local_idx]
            tp_c = ae_per_class_tp[local_idx]

            if total_c == 0:
                r = 0.0
            else:
                r = tp_c / total_c

            print(
                f"[{cls_name:<24}] total={total_c:5d} | "
                f"AE_TP={tp_c:5d} | AE_recall={r:.4f}"
            )


if __name__ == "__main__":
    with torch.no_grad():
        run_fusion_test()
