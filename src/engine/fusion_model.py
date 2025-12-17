# src/engine/fusion_model.py

from typing import Literal, Tuple
import torch
import numpy as np

AggMode = Literal["mean", "max", "topk"]


@torch.no_grad()
def compute_ae_scores(
    recon: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    agg_mode: AggMode = "topk",
    topk: int = 3,
) -> torch.Tensor:
    """
    AE 재구성 에러를 윈도우 단위 스코어로 집계.

    **주의: AutoEncoder/test.py 의 evaluate_ae() 스코어 정의와 1:1로 동일하게 맞춤.**

    recon  : [B, L, F]
    target : [B, L, F]
    mask   : [B, L]   (1=real, 0=pad)

    반환:
      scores: [B]  (윈도우별 스칼라 score)
    """
    device = recon.device

    # [B, L, 1]
    mask_exp = mask.unsqueeze(-1).float()

    # [B, L, F]  (padding 위치는 0)
    diff2 = ((recon - target) ** 2) * mask_exp

    # flow 단위 error: feature dimension "합" → [B, L]
    flow_err = diff2.sum(dim=2)  # test.py와 동일

    if agg_mode == "mean":
        # 모든 유효 element 평균
        # valid_counts: [B]  (mask==1 인 (L*F) 개수)
        valid_counts = mask_exp.sum(dim=(1, 2)) + 1e-8  # [B]
        # diff2.sum(dim=(1,2)) : [B]
        scores = diff2.sum(dim=(1, 2)) / valid_counts   # [B]

    elif agg_mode == "max":
        # flow_err * mask 해서, mask=1인 위치만 고려
        # max over time-step (L)
        # 결과: [B]
        scores, _ = (flow_err * mask).max(dim=1)

    elif agg_mode == "topk":
        B, L = flow_err.shape
        scores_list = []
        for b in range(B):
            valid = (mask[b] > 0)            # [L] bool
            fe = flow_err[b][valid]          # [M], feature-sum 기반
            if fe.numel() == 0:
                scores_list.append(torch.tensor(0.0, device=device))
            else:
                k_eff = min(topk, fe.numel())
                topk_vals, _ = fe.topk(k_eff)
                scores_list.append(topk_vals.mean())
        scores = torch.stack(scores_list, dim=0)  # [B]

    else:
        raise ValueError(f"Unknown agg_mode: {agg_mode}")

    return scores


def update_recall_stats(
    y_true: torch.Tensor,
    y_pred_tcn: torch.Tensor,
    ae_recovered_mask: torch.Tensor,
) -> Tuple[
    int,  # total_attack
    int,  # tcn_tp_total
    int,  # tcn_fn_total
    np.ndarray,  # per_class_total (attack classes만)
    np.ndarray,  # per_class_tcn_tp
    np.ndarray,  # per_class_ae_recover
]:

    """
    fusion_test 내부에서 호출하는 통계 집계용 유틸.
    label_classes[0] = 'BENIGN' 이고,
    1번 인덱스부터가 공격 클래스라는 전제를 사용.
    """
    # y_true, y_pred_tcn: [B, C] (multi-label 0/1)
    # ae_recovered_mask : [B] (True = AE가 이 윈도우를 공격으로 인지)

    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred_tcn.cpu().numpy()
    ae_rec_np = ae_recovered_mask.cpu().numpy()

    C = y_true_np.shape[1]
    attack_indices = list(range(1, C))  # 1~C-1

    per_class_total = np.zeros(len(attack_indices), dtype=np.int64)
    per_class_tcn_tp = np.zeros(len(attack_indices), dtype=np.int64)
    per_class_ae_recover = np.zeros(len(attack_indices), dtype=np.int64)

    # 공격 윈도우 (한 개라도 공격 라벨이 있는 윈도우)
    true_attack_any = (y_true_np[:, 1:] == 1).any(axis=1)  # [B]
    pred_attack_any = (y_pred_np[:, 1:] == 1).any(axis=1)  # [B]

    total_attack = int(true_attack_any.sum())
    tcn_tp_total = int((true_attack_any & pred_attack_any).sum())
    tcn_fn_total = int((true_attack_any & ~pred_attack_any).sum())

    # per-class 통계
    for local_idx, c in enumerate(attack_indices):
        true_c = (y_true_np[:, c] == 1)
        pred_c = (y_pred_np[:, c] == 1)

        per_class_total[local_idx] = int(true_c.sum())
        per_class_tcn_tp[local_idx] = int((true_c & pred_c).sum())
        # AE가 이 윈도우를 공격으로 본 것 중, 해당 클래스가 실제로 1인 윈도우 수
        per_class_ae_recover[local_idx] = int((true_c & ae_rec_np).sum())

    return (
        total_attack,
        tcn_tp_total,
        tcn_fn_total,
        per_class_total,
        per_class_tcn_tp,
        per_class_ae_recover,
    )
