#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LayoutLMv3 UI检测系统 - 消融实验

"""

import os
import csv
import json
import gc
import random
import warnings
from datetime import datetime

import torch
import numpy as np

from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from transformers import (
    LayoutLMv3Config,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    logging
)


# ==================== 0. 全局设置 ====================
warnings.filterwarnings("ignore")
logging.set_verbosity_info()

os.environ["WANDB_DISABLED"] = "true"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ==================== 1. 路径与实验配置 ====================
TRAIN_PATH = "/home/zupupdec/LayoutLMV3_Fine_Tuning/split_layoutlmv3_family_merged/train.json"
VAL_PATH = "/home/zupupdec/LayoutLMV3_Fine_Tuning/split_layoutlmv3_family_merged/validation.json"
TEST_PATH = "/home/zupupdec/LayoutLMV3_Fine_Tuning/split_layoutlmv3_family_merged/test.json"

BASE_MODEL_NAME = "microsoft/layoutlmv3-base"

RUN_ABLATIONS = [
    "wo_relation",
    "wo_geometry",
    "wo_ocr",
    "wo_conf_mask",
    "wo_pseudo_weak",
    "wo_ppl",
    "wo_ppl_and_relation"
]

# 如果要把完整模型也一起跑，可以改成：
# RUN_ABLATIONS = [
#     "none",
#     "wo_relation",
#     "wo_geometry",
#     "wo_ocr",
#     "wo_conf_mask",
#     "wo_pseudo_weak",
#     "wo_ppl",
#     "wo_ppl_and_relation"
# ]

NUM_EPOCHS = 30
REPORT_EVERY = 3
EVAL_EPOCHS = list(range(REPORT_EVERY, NUM_EPOCHS + 1, REPORT_EVERY))

PREDICTION_THRESHOLD = 0.5
MAX_LENGTH = 512

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16

SEED = 42

RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_ROOT = f"./ablation_auto_results_{RUN_TAG}"

LONG_CSV = os.path.join(
    RESULT_ROOT,
    "ablation_metrics_long_every3epochs.csv"
)

WIDE_CSV = os.path.join(
    RESULT_ROOT,
    "ablation_metrics_wide_every3epochs.csv"
)

FINAL_TEST_CSV = os.path.join(
    RESULT_ROOT,
    "ablation_epoch30_test_wide_summary.csv"
)


# ==================== 2. 随机种子 ====================
def set_all_seeds(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_cuda():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==================== 3. 动态规则判决器 ====================
class GlobalConfig:
    COVERAGE = 0.05
    DIST = 50.0
    HIERARCHY_DEFECT = 5.0
    MISALIGN_SCORE = 15.0


class UIHeuristics:
    @staticmethod
    def calculate_raw_metrics(bboxes):
        valid_boxes = [
            b for b in bboxes
            if isinstance(b, (list, tuple))
            and len(b) == 4
            and b[2] > b[0]
            and b[3] > b[1]
        ]

        count = len(valid_boxes)

        if count < 3:
            return 0.0, 1000.0, 0.0, 0.0

        total_area = sum(
            [
                (b[2] - b[0]) * (b[3] - b[1])
                for b in valid_boxes
            ]
        )

        coverage = total_area / 1000000.0

        centers = np.array(
            [
                [
                    (b[0] + b[2]) / 2.0,
                    (b[1] + b[3]) / 2.0
                ]
                for b in valid_boxes
            ]
        )

        min_dists = []

        for i in range(count):
            dists = np.sum(
                np.abs(centers - centers[i]),
                axis=1
            )

            dists[i] = 99999
            min_dists.append(np.min(dists))

        avg_dist = np.mean(min_dists) if min_dists else 1000.0

        centers_y = [
            (b[1] + b[3]) / 2.0
            for b in valid_boxes
        ]

        heights = [
            b[3] - b[1]
            for b in valid_boxes
        ]

        sorted_indices = np.argsort(centers_y)

        rows = []
        current_row = [sorted_indices[0]]

        for idx in sorted_indices[1:]:
            if abs(centers_y[idx] - centers_y[current_row[-1]]) < 20:
                current_row.append(idx)
            else:
                rows.append(current_row)
                current_row = [idx]

        if current_row:
            rows.append(current_row)

        local_cvs = []

        for row in rows:
            if len(row) >= 2:
                row_h = [
                    heights[i]
                    for i in row
                ]

                local_cvs.append(
                    np.std(row_h) / (np.mean(row_h) + 1e-5)
                )

        avg_local_cv = np.mean(local_cvs) if local_cvs else 0.0

        areas = sorted(
            [
                (b[2] - b[0]) * (b[3] - b[1])
                for b in valid_boxes
            ],
            reverse=True
        )

        top_10_idx = max(1, len(areas) // 10)
        bottom_50_idx = max(1, len(areas) // 2)

        top_area_avg = np.mean(areas[:top_10_idx])
        bottom_area_avg = np.mean(areas[-bottom_50_idx:])

        dominance_ratio = top_area_avg / (
            bottom_area_avg + 1e-5
        )

        hierarchy_defect = (
            avg_local_cv * 10.0
        ) + (
            10.0 / (dominance_ratio + 1.0)
        )

        def get_jitter(coords):
            if not coords:
                return 0.0

            coords_sorted = sorted(coords)
            jitter = 0.0
            groups = [[coords_sorted[0]]]

            for x in coords_sorted[1:]:
                if x - groups[-1][-1] < 20:
                    groups[-1].append(x)
                else:
                    groups.append([x])

            for g in groups:
                if len(g) >= 3:
                    jitter += np.std(g)

            return jitter

        jitter_left = get_jitter(
            [
                b[0]
                for b in valid_boxes
            ]
        )

        jitter_right = get_jitter(
            [
                b[2]
                for b in valid_boxes
            ]
        )

        jitter_center = get_jitter(
            [
                (b[0] + b[2]) / 2.0
                for b in valid_boxes
            ]
        )

        misalign_score = (
            jitter_left + jitter_right + jitter_center
        ) / 3.0

        return coverage, avg_dist, hierarchy_defect, misalign_score

    @staticmethod
    def get_continuous_labels(bboxes):
        cov, dist, h_defect, m_score = UIHeuristics.calculate_raw_metrics(bboxes)

        cov_thresh = GlobalConfig.COVERAGE + 1e-5
        dist_thresh = GlobalConfig.DIST
        m_thresh = GlobalConfig.MISALIGN_SCORE + 1e-5
        h_thresh = GlobalConfig.HIERARCHY_DEFECT + 1e-5

        def sigmoid_mapping(val, thresh, reverse=False, T=10.0):
            if reverse:
                diff_ratio = (thresh - val) / thresh
            else:
                diff_ratio = (val - thresh) / thresh

            diff_ratio = max(min(diff_ratio, 5.0), -5.0)

            return 1.0 / (
                1.0 + np.exp(-diff_ratio * T)
            )

        p_cov = sigmoid_mapping(
            cov,
            cov_thresh,
            reverse=False
        )

        p_dist = sigmoid_mapping(
            dist,
            dist_thresh,
            reverse=True
        )

        score_overcrowd = max(p_cov, p_dist)

        score_misalign = sigmoid_mapping(
            m_score,
            m_thresh,
            reverse=False
        )

        score_hierarchy = sigmoid_mapping(
            h_defect,
            h_thresh,
            reverse=False
        )

        return [
            score_overcrowd,
            score_misalign,
            score_hierarchy
        ]


# ==================== 4. bbox 归一化与训练集校准 ====================
def normalize_boxes_for_calibration(bboxes):
    if not bboxes:
        return []

    valid = [
        b for b in bboxes
        if isinstance(b, (list, tuple))
        and len(b) == 4
    ]

    if not valid:
        return []

    max_x = max(
        [
            b[2]
            for b in valid
        ] + [1]
    )

    max_y = max(
        [
            b[3]
            for b in valid
        ] + [1]
    )

    scale_x = 1000.0 / max_x if max_x > 1000 else 1.0
    scale_y = 1000.0 / max_y if max_y > 1000 else 1.0

    new_bboxes = []

    for b in valid:
        x1 = min(max(0, int(b[0] * scale_x)), 1000)
        y1 = min(max(0, int(b[1] * scale_y)), 1000)
        x2 = min(max(0, int(b[2] * scale_x)), 1000)
        y2 = min(max(0, int(b[3] * scale_y)), 1000)

        if x2 <= x1:
            x2 = min(x1 + 1, 1000)

        if y2 <= y1:
            y2 = min(y1 + 1, 1000)

        new_bboxes.append(
            [
                x1,
                y1,
                x2,
                y2
            ]
        )

    return new_bboxes


def calibrate_dataset_quantile(train_data):
    print("\n🚀 [智能校准] 正在扫描训练集原始数据...")

    all_covs = []
    all_dists = []
    all_hdefs = []
    all_mscores = []

    for item in train_data:
        bboxes = normalize_boxes_for_calibration(
            item.get("bbox", [])
        )

        cov, dist, h_defect, m_score = UIHeuristics.calculate_raw_metrics(bboxes)

        all_covs.append(cov)
        all_dists.append(dist)
        all_hdefs.append(h_defect)
        all_mscores.append(m_score)

    GlobalConfig.COVERAGE = float(
        np.percentile(all_covs, 75)
    )

    GlobalConfig.DIST = float(
        np.percentile(all_dists, 25)
    )

    GlobalConfig.HIERARCHY_DEFECT = float(
        np.percentile(all_hdefs, 75)
    )

    GlobalConfig.MISALIGN_SCORE = max(
        0.1,
        float(np.percentile(all_mscores, 75))
    )

    print("🎯 基于训练集的校准完成！")
    print(f"COVERAGE         = {GlobalConfig.COVERAGE:.6f}")
    print(f"DIST             = {GlobalConfig.DIST:.6f}")
    print(f"HIERARCHY_DEFECT = {GlobalConfig.HIERARCHY_DEFECT:.6f}")
    print(f"MISALIGN_SCORE   = {GlobalConfig.MISALIGN_SCORE:.6f}")


# ==================== 5. 数据集 ====================
class RealDataset(Dataset):
    def __init__(self, raw_data, is_train=False, ablation="none"):
        self.raw_data = raw_data
        self.is_train = is_train
        self.ablation = ablation

    def _pad(self, seq, val=0):
        if len(seq) > MAX_LENGTH:
            return seq[:MAX_LENGTH]

        return seq + [val] * (
            MAX_LENGTH - len(seq)
        )

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        item = self.raw_data[idx]

        bboxes = item.get(
            "bbox",
            []
        )

        input_ids = item.get(
            "input_ids",
            []
        )

        # 消融：w/o OCR / text tokens
        if self.ablation == "wo_ocr":
            input_ids = []

        norm_boxes = normalize_boxes_for_calibration(
            bboxes
        )

        continuous_label = UIHeuristics.get_continuous_labels(
            norm_boxes
        )

        # 关键改动：
        # wo_pseudo_weak 不再使用粗糙随机弱标签。
        # 改为：只在训练集把连续软标签直接二值化为 0/1 硬标签。
        if self.ablation == "wo_pseudo_weak" and self.is_train:
            final_label = [
                1.0 if x >= 0.5 else 0.0
                for x in continuous_label
            ]
        else:
            final_label = continuous_label

        # 防止极少数样本没有有效 bbox
        if len(norm_boxes) == 0:
            model_boxes = [[0, 0, 1, 1]]
        else:
            model_boxes = norm_boxes

        valid_len = min(
            len(model_boxes),
            MAX_LENGTH
        )

        return {
            "input_ids": torch.tensor(
                self._pad(input_ids, 0),
                dtype=torch.long
            ),

            "bbox": torch.tensor(
                self._pad(model_boxes, [0, 0, 0, 0]),
                dtype=torch.long
            ),

            "attention_mask": torch.tensor(
                self._pad([1] * valid_len, 0),
                dtype=torch.long
            ),

            "labels": torch.tensor(
                final_label,
                dtype=torch.float32
            )
        }


# ==================== 6. 核心网络 ====================
class ParallelPerceptionLayer(nn.Module):
    def __init__(self, hidden_size=768, ablation="none"):
        super().__init__()

        self.ablation = ablation

        # w/o geometry features：只输入原始 bbox 四维
        in_dim = 4 if self.ablation == "wo_geometry" else 8

        self.bbox_encoder = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, hidden_size),
            nn.LayerNorm(hidden_size)
        )

        self.num_heads = 8

        spatial_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=self.num_heads,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )

        self.spatial_encoder = nn.TransformerEncoder(
            spatial_layer,
            num_layers=2
        )

        semantic_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=self.num_heads,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )

        self.semantic_encoder = nn.TransformerEncoder(
            semantic_layer,
            num_layers=1
        )

        self.relation_proj = nn.Linear(
            11,
            self.num_heads,
            bias=False
        )

        self.gate_overcrowd = nn.Sequential(
            nn.Linear(hidden_size * 6, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        self.gate_misalign = nn.Sequential(
            nn.Linear(hidden_size * 6, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        self.gate_hierarchy = nn.Sequential(
            nn.Linear(hidden_size * 6, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def compute_relation_bias(self, norm_bboxes):
        B, N, _ = norm_bboxes.shape

        w = norm_bboxes[:, :, 2] - norm_bboxes[:, :, 0]
        h = norm_bboxes[:, :, 3] - norm_bboxes[:, :, 1]

        cx = (
            norm_bboxes[:, :, 0]
            + norm_bboxes[:, :, 2]
        ) / 2.0

        cy = (
            norm_bboxes[:, :, 1]
            + norm_bboxes[:, :, 3]
        ) / 2.0

        area = w * h

        dx = cx.unsqueeze(2) - cx.unsqueeze(1)
        dy = cy.unsqueeze(2) - cy.unsqueeze(1)

        w_ratio = torch.log(
            (w.unsqueeze(2) + 1e-5)
            / (w.unsqueeze(1) + 1e-5)
        )

        h_ratio = torch.log(
            (h.unsqueeze(2) + 1e-5)
            / (h.unsqueeze(1) + 1e-5)
        )

        area_ratio = torch.log(
            (area.unsqueeze(2) + 1e-5)
            / (area.unsqueeze(1) + 1e-5)
        )

        left_gap = (
            norm_bboxes[:, :, 0].unsqueeze(2)
            - norm_bboxes[:, :, 0].unsqueeze(1)
        )

        right_gap = (
            norm_bboxes[:, :, 2].unsqueeze(2)
            - norm_bboxes[:, :, 2].unsqueeze(1)
        )

        center_gap = torch.sqrt(
            dx ** 2 + dy ** 2 + 1e-6
        )

        same_row = (
            torch.abs(dy) < 0.02
        ).float()

        same_col = (
            torch.abs(dx) < 0.02
        ).float()

        b1 = norm_bboxes.unsqueeze(2)
        b2 = norm_bboxes.unsqueeze(1)

        inter_lt = torch.max(
            b1[..., :2],
            b2[..., :2]
        )

        inter_rb = torch.min(
            b1[..., 2:],
            b2[..., 2:]
        )

        inter_wh = torch.clamp(
            inter_rb - inter_lt,
            min=0
        )

        inter_area = (
            inter_wh[..., 0]
            * inter_wh[..., 1]
        )

        iou = inter_area / (
            area.unsqueeze(2)
            + area.unsqueeze(1)
            - inter_area
            + 1e-6
        )

        r_ij = torch.stack(
            [
                dx,
                dy,
                w_ratio,
                h_ratio,
                area_ratio,
                left_gap,
                right_gap,
                center_gap,
                same_row,
                same_col,
                iou
            ],
            dim=-1
        )

        rel_bias = self.relation_proj(
            r_ij
        )

        rel_bias = rel_bias.permute(
            0,
            3,
            1,
            2
        ).reshape(
            B * self.num_heads,
            N,
            N
        )

        return rel_bias

    def forward(self, hidden_states, bboxes, attention_mask):
        norm_bboxes = bboxes.float() / 1000.0

        w = norm_bboxes[:, :, 2] - norm_bboxes[:, :, 0]
        h = norm_bboxes[:, :, 3] - norm_bboxes[:, :, 1]

        cx = (
            norm_bboxes[:, :, 0]
            + norm_bboxes[:, :, 2]
        ) / 2.0

        cy = (
            norm_bboxes[:, :, 1]
            + norm_bboxes[:, :, 3]
        ) / 2.0

        if self.ablation == "wo_geometry":
            enhanced_bboxes = norm_bboxes
        else:
            enhanced_bboxes = torch.cat(
                [
                    norm_bboxes,
                    w.unsqueeze(-1),
                    h.unsqueeze(-1),
                    cx.unsqueeze(-1),
                    cy.unsqueeze(-1)
                ],
                dim=-1
            )

        padding_mask = attention_mask == 0

        # w/o relation-aware attention bias
        if self.ablation == "wo_relation":
            relation_bias = None
        else:
            relation_bias = self.compute_relation_bias(
                norm_bboxes
            )

        spatial_features = self.spatial_encoder(
            self.bbox_encoder(enhanced_bboxes),
            mask=relation_bias,
            src_key_padding_mask=padding_mask
        )

        semantic_features = self.semantic_encoder(
            hidden_states,
            src_key_padding_mask=padding_mask
        )

        mask = attention_mask.unsqueeze(-1).bool()
        mask_float = attention_mask.unsqueeze(-1).float()

        spatial_features_masked = spatial_features.masked_fill(
            ~mask,
            0.0
        )

        semantic_features_masked = semantic_features.masked_fill(
            ~mask,
            0.0
        )

        spatial_pool_max = spatial_features.masked_fill(
            ~mask,
            -1e9
        ).max(dim=1)[0]

        semantic_pool_max = semantic_features.masked_fill(
            ~mask,
            -1e9
        ).max(dim=1)[0]

        spatial_pool_mean = spatial_features_masked.sum(dim=1) / (
            mask_float.sum(dim=1) + 1e-9
        )

        semantic_pool_mean = semantic_features_masked.sum(dim=1) / (
            mask_float.sum(dim=1) + 1e-9
        )

        spatial_diff_sq = (
            spatial_features
            - spatial_pool_mean.unsqueeze(1)
        ) ** 2

        spatial_pool_std = torch.sqrt(
            spatial_diff_sq.masked_fill(
                ~mask,
                0.0
            ).sum(dim=1) / (
                mask_float.sum(dim=1) + 1e-9
            ) + 1e-6
        )

        semantic_diff_sq = (
            semantic_features
            - semantic_pool_mean.unsqueeze(1)
        ) ** 2

        semantic_pool_std = torch.sqrt(
            semantic_diff_sq.masked_fill(
                ~mask,
                0.0
            ).sum(dim=1) / (
                mask_float.sum(dim=1) + 1e-9
            ) + 1e-6
        )

        spatial_pool = torch.cat(
            [
                spatial_pool_max,
                spatial_pool_mean,
                spatial_pool_std
            ],
            dim=-1
        )

        semantic_pool = torch.cat(
            [
                semantic_pool_max,
                semantic_pool_mean,
                semantic_pool_std
            ],
            dim=-1
        )

        concat_pool = torch.cat(
            [
                spatial_pool,
                semantic_pool
            ],
            dim=-1
        )

        weight_overcrowd = F.softmax(
            self.gate_overcrowd(concat_pool),
            dim=-1
        )

        weight_misalign = F.softmax(
            self.gate_misalign(concat_pool),
            dim=-1
        )

        weight_hierarchy = F.softmax(
            self.gate_hierarchy(concat_pool),
            dim=-1
        )

        feat_overcrowd = (
            weight_overcrowd[:, 0:1] * spatial_pool
            + weight_overcrowd[:, 1:2] * semantic_pool
        )

        feat_misalign = (
            weight_misalign[:, 0:1] * spatial_pool
            + weight_misalign[:, 1:2] * semantic_pool
        )

        feat_hierarchy = (
            weight_hierarchy[:, 0:1] * spatial_pool
            + weight_hierarchy[:, 1:2] * semantic_pool
        )

        return feat_overcrowd, feat_misalign, feat_hierarchy


class FinalModel(LayoutLMv3ForTokenClassification):
    def __init__(self, config, ablation="none"):
        super().__init__(config)

        self.ablation = ablation

        hidden_size = config.hidden_size if hasattr(
            config,
            "hidden_size"
        ) else 768

        # w/o PPL / w/o PPL+RAB：不使用 Parallel Perception Layer，
        # 直接使用 LayoutLMv3 最后一层的 first-token / [CLS] 表征。
        if self.ablation not in [
            "wo_ppl",
            "wo_ppl_and_relation"
        ]:
            self.perception_layer = ParallelPerceptionLayer(
                hidden_size,
                ablation
            )
            head_in_dim = hidden_size * 3
        else:
            head_in_dim = hidden_size

        # 三个独立分类头，分别对应三类缺陷。
        self.head_overcrowd = nn.Sequential(
            nn.Linear(head_in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        self.head_misalign = nn.Sequential(
            nn.Linear(head_in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        self.head_hierarchy = nn.Sequential(
            nn.Linear(head_in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        self.loss_fct = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                [1.0, 1.0, 1.0],
                dtype=torch.float32
            ),
            reduction="none"
        )

    def forward(self, input_ids=None, bbox=None, attention_mask=None, labels=None, **kwargs):
        if self.loss_fct.pos_weight.device != input_ids.device:
            self.loss_fct.pos_weight = self.loss_fct.pos_weight.to(input_ids.device)

        outputs = self.layoutlmv3(
            input_ids=input_ids,
            bbox=bbox.clamp(0, 1000),
            attention_mask=attention_mask
        )

        if self.ablation in [
            "wo_ppl",
            "wo_ppl_and_relation"
        ]:
            cls_feat = outputs.last_hidden_state[:, 0, :]

            logit_o = self.head_overcrowd(
                cls_feat
            )

            logit_m = self.head_misalign(
                cls_feat
            )

            logit_h = self.head_hierarchy(
                cls_feat
            )

        else:
            feat_overcrowd, feat_misalign, feat_hierarchy = self.perception_layer(
                hidden_states=outputs.last_hidden_state,
                bboxes=bbox,
                attention_mask=attention_mask
            )

            logit_o = self.head_overcrowd(
                feat_overcrowd
            )

            logit_m = self.head_misalign(
                feat_misalign
            )

            logit_h = self.head_hierarchy(
                feat_hierarchy
            )

        logits = torch.cat(
            [
                logit_o,
                logit_m,
                logit_h
            ],
            dim=-1
        )

        loss = None

        if labels is not None:
            raw_loss = self.loss_fct(
                logits,
                labels.float()
            )

            # w/o confidence mask：不使用置信度遮罩。
            # wo_pseudo_weak：训练标签已经是 0/1 硬标签，也不需要置信度遮罩。
            if self.ablation in [
                "wo_conf_mask",
                "wo_pseudo_weak"
            ]:
                loss = raw_loss.mean()

            else:
                # 排除 [0.45, 0.55] 区间内的不确定软标签。
                confident_mask = (
                    (labels < 0.45)
                    | (labels > 0.55)
                ).float()

                loss = (
                    raw_loss * confident_mask
                ).sum() / (
                    confident_mask.sum() + 1e-9
                )

        return (loss, logits) if loss is not None else (logits,)


# ==================== 7. 指标计算与保存 ====================
def sigmoid_np(x):
    return 1.0 / (
        1.0 + np.exp(-np.clip(x, -50, 50))
    )


def compute_metric_dict(predictions, label_ids):
    probs = sigmoid_np(predictions)

    true_binary = (
        label_ids > 0.5
    ).astype(int)

    pred_binary = (
        probs > PREDICTION_THRESHOLD
    ).astype(int)

    tasks = [
        ("overcrowding", "Overcrowding", 0),
        ("misalignment", "Misalignment", 1),
        ("hierarchy", "Poor Visual Hierarchy", 2)
    ]

    metrics = {}

    macro_auc = 0.0
    macro_auprc = 0.0
    macro_p = 0.0
    macro_r = 0.0
    macro_f1 = 0.0

    for key, name, i in tasks:
        y_true = true_binary[:, i]
        y_score = probs[:, i]
        y_pred = pred_binary[:, i]

        support = int(
            np.sum(y_true)
        )

        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(
                y_true,
                y_score
            )

            auprc = average_precision_score(
                y_true,
                y_score
            )
        else:
            auc = 0.5
            auprc = 1.0 if support > 0 else 0.0

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        metrics[key] = {
            "name": name,
            "auc": auc,
            "auprc": auprc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support
        }

        macro_auc += auc
        macro_auprc += auprc
        macro_p += precision
        macro_r += recall
        macro_f1 += f1

    metrics["macro"] = {
        "name": "Macro Avg",
        "auc": macro_auc / 3.0,
        "auprc": macro_auprc / 3.0,
        "precision": macro_p / 3.0,
        "recall": macro_r / 3.0,
        "f1": macro_f1 / 3.0,
        "support": ""
    }

    return metrics


def make_long_rows(metrics, ablation, epoch, split_name):
    rows = []

    for key in [
        "overcrowding",
        "misalignment",
        "hierarchy",
        "macro"
    ]:
        m = metrics[key]

        rows.append(
            {
                "run_tag": RUN_TAG,
                "ablation": ablation,
                "epoch": epoch,
                "split": split_name,
                "defect_type": m["name"],
                "auc": round(m["auc"], 3),
                "auprc": round(m["auprc"], 3),
                "precision": round(m["precision"], 3),
                "recall": round(m["recall"], 3),
                "f1": round(m["f1"], 3),
                "support": m["support"]
            }
        )

    return rows


def make_wide_row(metrics, ablation, epoch, split_name):
    row = {
        "run_tag": RUN_TAG,
        "ablation": ablation,
        "epoch": epoch,
        "split": split_name
    }

    for key in [
        "overcrowding",
        "misalignment",
        "hierarchy"
    ]:
        m = metrics[key]

        row[f"{key}_p"] = round(m["precision"], 3)
        row[f"{key}_r"] = round(m["recall"], 3)
        row[f"{key}_f1"] = round(m["f1"], 3)
        row[f"{key}_auc"] = round(m["auc"], 3)
        row[f"{key}_auprc"] = round(m["auprc"], 3)
        row[f"{key}_support"] = m["support"]

    row["macro_p"] = round(
        metrics["macro"]["precision"],
        3
    )

    row["macro_r"] = round(
        metrics["macro"]["recall"],
        3
    )

    row["macro_f1"] = round(
        metrics["macro"]["f1"],
        3
    )

    row["macro_auc"] = round(
        metrics["macro"]["auc"],
        3
    )

    row["macro_auprc"] = round(
        metrics["macro"]["auprc"],
        3
    )

    return row


def print_metric_table(metrics, title):
    print(f"\n{'=' * 115}")
    print(title)
    print(f"{'-' * 115}")
    print(
        f"{'Defect Type':<22} | "
        f"{'AUC':<7} | "
        f"{'AUPRC':<7} | "
        f"{'P':<7} | "
        f"{'R':<7} | "
        f"{'F1':<7} | "
        f"{'Support':<8}"
    )
    print(f"{'-' * 115}")

    for key in [
        "overcrowding",
        "misalignment",
        "hierarchy",
        "macro"
    ]:
        m = metrics[key]

        print(
            f"{m['name']:<22} | "
            f"{m['auc']:.3f}   | "
            f"{m['auprc']:.3f}   | "
            f"{m['precision']:.3f}   | "
            f"{m['recall']:.3f}   | "
            f"{m['f1']:.3f}   | "
            f"{str(m['support']):<8}"
        )

    print(f"{'=' * 115}\n")


def append_dicts_to_csv(rows, csv_path, fieldnames):
    os.makedirs(
        os.path.dirname(csv_path),
        exist_ok=True
    )

    file_exists = os.path.exists(csv_path)

    with open(
        csv_path,
        "a",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)


LONG_FIELDNAMES = [
    "run_tag",
    "ablation",
    "epoch",
    "split",
    "defect_type",
    "auc",
    "auprc",
    "precision",
    "recall",
    "f1",
    "support"
]

WIDE_FIELDNAMES = [
    "run_tag",
    "ablation",
    "epoch",
    "split",

    "overcrowding_p",
    "overcrowding_r",
    "overcrowding_f1",
    "overcrowding_auc",
    "overcrowding_auprc",
    "overcrowding_support",

    "misalignment_p",
    "misalignment_r",
    "misalignment_f1",
    "misalignment_auc",
    "misalignment_auprc",
    "misalignment_support",

    "hierarchy_p",
    "hierarchy_r",
    "hierarchy_f1",
    "hierarchy_auc",
    "hierarchy_auprc",
    "hierarchy_support",

    "macro_p",
    "macro_r",
    "macro_f1",
    "macro_auc",
    "macro_auprc"
]


# ==================== 8. Callback：每个消融每 3 轮保存 Validation + Test ====================
class SaveMetricsEveryNEpochsCallback(TrainerCallback):
    def __init__(
        self,
        val_dataset,
        test_dataset,
        ablation_name,
        every_n_epochs=3
    ):
        self.trainer = None
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.ablation_name = ablation_name
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(
            round(state.epoch)
        )

        if epoch <= 0:
            return control

        if epoch % self.every_n_epochs != 0:
            print(
                f"\n📌 {self.ablation_name} | "
                f"Epoch {epoch} finished. Skip saving."
            )
            return control

        print("\n" + "#" * 120)
        print(
            f"📌 {self.ablation_name.upper()} | "
            f"Epoch {epoch} | 保存 Validation + Test 指标"
        )
        print("#" * 120)

        for split_name, dataset in [
            ("validation", self.val_dataset),
            ("test", self.test_dataset)
        ]:
            pred_output = self.trainer.predict(
                dataset
            )

            metrics = compute_metric_dict(
                predictions=pred_output.predictions,
                label_ids=pred_output.label_ids
            )

            print_metric_table(
                metrics,
                title=(
                    f"{self.ablation_name.upper()} | "
                    f"{split_name.upper()} | "
                    f"Epoch {epoch}"
                )
            )

            long_rows = make_long_rows(
                metrics=metrics,
                ablation=self.ablation_name,
                epoch=epoch,
                split_name=split_name
            )

            wide_row = make_wide_row(
                metrics=metrics,
                ablation=self.ablation_name,
                epoch=epoch,
                split_name=split_name
            )

            append_dicts_to_csv(
                long_rows,
                LONG_CSV,
                LONG_FIELDNAMES
            )

            append_dicts_to_csv(
                [wide_row],
                WIDE_CSV,
                WIDE_FIELDNAMES
            )

            if epoch == NUM_EPOCHS and split_name == "test":
                append_dicts_to_csv(
                    [wide_row],
                    FINAL_TEST_CSV,
                    WIDE_FIELDNAMES
                )

        print(
            f"✅ {self.ablation_name.upper()} | "
            f"Epoch {epoch} 指标已保存"
        )

        print(f"📄 Long CSV : {LONG_CSV}")
        print(f"📄 Wide CSV : {WIDE_CSV}")

        return control


# ==================== 9. 模型构建与训练 ====================
def build_model(ablation_name):
    config = LayoutLMv3Config.from_pretrained(
        BASE_MODEL_NAME,
        local_files_only=True
    )

    model = FinalModel.from_pretrained(
        BASE_MODEL_NAME,
        config=config,
        ablation=ablation_name,
        local_files_only=True,
        ignore_mismatched_sizes=True
    )

    if torch.cuda.is_available():
        model = model.cuda()

    # 冻结 LayoutLMv3 前 8 层，只解冻后 4 层
    for name, param in model.layoutlmv3.named_parameters():
        if any(
            f"layer.{i}" in name
            for i in range(8, 12)
        ):
            param.requires_grad = True
        else:
            param.requires_grad = False

    # PPL 存在时解冻 PPL
    if hasattr(model, "perception_layer"):
        for param in model.perception_layer.parameters():
            param.requires_grad = True

    # 解冻所有 head
    for name, param in model.named_parameters():
        if "head" in name:
            param.requires_grad = True

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"可训练参数量: {trainable_params:,}")

    return model


def build_optimizer(model):
    optimizer_grouped_parameters = []

    layoutlmv3_params = [
        p for n, p in model.named_parameters()
        if "layoutlmv3" in n and p.requires_grad
    ]

    if layoutlmv3_params:
        optimizer_grouped_parameters.append(
            {
                "params": layoutlmv3_params,
                "lr": 2e-5
            }
        )

    if hasattr(model, "perception_layer"):
        perception_params = [
            p for n, p in model.named_parameters()
            if "perception_layer" in n and p.requires_grad
        ]

        if perception_params:
            optimizer_grouped_parameters.append(
                {
                    "params": perception_params,
                    "lr": 1e-4
                }
            )

    head_params = [
        p for n, p in model.named_parameters()
        if "head" in n and p.requires_grad
    ]

    if head_params:
        optimizer_grouped_parameters.append(
            {
                "params": head_params,
                "lr": 1e-4
            }
        )

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        weight_decay=0.01
    )

    return optimizer


def run_one_ablation(
    ablation_name,
    train_data,
    val_data,
    test_data
):
    print("\n" + "=" * 120)
    print(f"🚀 开始消融实验：{ablation_name.upper()}")
    print("=" * 120)

    print(
        f"📌 当前消融 {ablation_name.upper()} 的保存轮次："
        f"{EVAL_EPOCHS}"
    )

    set_all_seeds(SEED)
    cleanup_cuda()

    train_set = RealDataset(
        train_data,
        is_train=True,
        ablation=ablation_name
    )

    val_set = RealDataset(
        val_data,
        is_train=False,
        ablation=ablation_name
    )

    test_set = RealDataset(
        test_data,
        is_train=False,
        ablation=ablation_name
    )

    if ablation_name == "wo_pseudo_weak":
        print(
            "🚨 当前为 wo_pseudo_weak："
            "训练集将连续软标签直接二值化为 0/1 硬标签；"
            "Validation/Test 仍使用统一连续标签并在评估时 >0.5 二值化。"
        )

    print(
        f"📦 数据集 | "
        f"Train: {len(train_data)} | "
        f"Validation: {len(val_data)} | "
        f"Test: {len(test_data)}"
    )

    model = build_model(
        ablation_name
    )

    optimizer = build_optimizer(
        model
    )

    callback = SaveMetricsEveryNEpochsCallback(
        val_dataset=val_set,
        test_dataset=test_set,
        ablation_name=ablation_name,
        every_n_epochs=REPORT_EVERY
    )

    trainer = Trainer(
        model=model,

        args=TrainingArguments(
            output_dir=os.path.join(
                RESULT_ROOT,
                "checkpoints",
                ablation_name
            ),

            num_train_epochs=NUM_EPOCHS,

            learning_rate=2e-5,

            lr_scheduler_type="cosine",

            warmup_ratio=0.1,

            per_device_train_batch_size=TRAIN_BATCH_SIZE,

            per_device_eval_batch_size=EVAL_BATCH_SIZE,

            dataloader_num_workers=4,

            # 关闭默认每轮评估，改为 callback 每 3 轮保存
            evaluation_strategy="no",

            logging_steps=10,

            save_strategy="no",

            fp16=torch.cuda.is_available(),

            report_to=[],

            remove_unused_columns=False
        ),

        train_dataset=train_set,

        eval_dataset=None,

        optimizers=(optimizer, None),

        compute_metrics=None,

        callbacks=[callback]
    )

    callback.trainer = trainer

    print(
        f"🚀 开始训练 {ablation_name.upper()} | "
        f"共 {NUM_EPOCHS} 轮 | "
        f"保存轮次 {EVAL_EPOCHS}"
    )

    trainer.train()

    print(f"✅ {ablation_name.upper()} 训练完成。")

    del trainer
    del model
    del optimizer

    cleanup_cuda()


# ==================== 10. Main ====================
def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def save_run_config(train_data, val_data, test_data):
    config_path = os.path.join(
        RESULT_ROOT,
        "run_config.json"
    )

    run_config = {
        "run_tag": RUN_TAG,
        "train_path": TRAIN_PATH,
        "val_path": VAL_PATH,
        "test_path": TEST_PATH,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "ablations": RUN_ABLATIONS,
        "num_epochs": NUM_EPOCHS,
        "report_every": REPORT_EVERY,
        "eval_epochs": EVAL_EPOCHS,
        "threshold": PREDICTION_THRESHOLD,
        "long_csv": LONG_CSV,
        "wide_csv": WIDE_CSV,
        "final_test_csv": FINAL_TEST_CSV,
        "wo_pseudo_weak_definition": (
            "training continuous soft labels are binarized into 0/1 hard labels; "
            "validation/test keep continuous labels and are binarized only for metric computation"
        )
    }

    with open(
        config_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            run_config,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"📄 运行配置已保存: {config_path}")


def main():
    os.makedirs(
        RESULT_ROOT,
        exist_ok=True
    )

    set_all_seeds(SEED)

    for p in [
        TRAIN_PATH,
        VAL_PATH,
        TEST_PATH
    ]:
        if not os.path.exists(p):
            print(f"❌ Json not found: {p}")
            return

    train_data = load_json(
        TRAIN_PATH
    )

    val_data = load_json(
        VAL_PATH
    )

    test_data = load_json(
        TEST_PATH
    )

    print("\n" + "=" * 120)
    print("📌 自动化消融实验启动")
    print("=" * 120)

    print(
        f"📦 固定数据划分 | "
        f"Train: {len(train_data)} | "
        f"Validation: {len(val_data)} | "
        f"Test: {len(test_data)}"
    )

    print(
        f"📌 每个消融都会在这些 epoch 保存 Validation + Test："
        f"{EVAL_EPOCHS}"
    )

    print(
        f"📌 本次将依次运行：{RUN_ABLATIONS}"
    )

    # 关键：只使用训练集进行软标签阈值校准
    calibrate_dataset_quantile(
        train_data
    )

    save_run_config(
        train_data,
        val_data,
        test_data
    )

    for ablation_name in RUN_ABLATIONS:
        run_one_ablation(
            ablation_name=ablation_name,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data
        )

    print("\n" + "=" * 120)
    print("🎉 所有消融实验已经全部运行完成")
    print("=" * 120)

    print(f"📄 Long-format 指标文件: {LONG_CSV}")
    print(f"📄 Wide-format 指标文件: {WIDE_CSV}")
    print(f"📄 Epoch 30 Test 汇总文件: {FINAL_TEST_CSV}")


if __name__ == "__main__":
    main()
