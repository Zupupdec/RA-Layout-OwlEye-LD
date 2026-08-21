#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LayoutLMv3 UI检测系统 - 主实验多标签版 (Multi-Label Classification)

"""

import os
import json
import torch
import numpy as np
import random
import warnings

from torch import nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import Dataset
from transformers import (
    LayoutLMv3Config,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    logging
)


# ==================== 0. 全局设置 ====================
warnings.filterwarnings("ignore")
logging.set_verbosity_info()

os.environ["WANDB_DISABLED"] = "true"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ==================== 1. 动态规则判决器 ====================
class GlobalConfig:
    COVERAGE = 0.05
    DIST = 50.0
    HIERARCHY_DEFECT = 5.0
    MISALIGN_SCORE = 15.0


class UIHeuristics:
    @staticmethod
    def calculate_raw_metrics(bboxes):
        valid_boxes = [b for b in bboxes if b[2] > b[0] and b[3] > b[1]]
        count = len(valid_boxes)

        if count < 3:
            return 0.0, 1000.0, 0.0, 0.0

        total_area = sum([(b[2] - b[0]) * (b[3] - b[1]) for b in valid_boxes])
        coverage = total_area / 1000000.0

        centers = np.array([
            [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]
            for b in valid_boxes
        ])

        min_dists = []
        for i in range(count):
            dists = np.sum(np.abs(centers - centers[i]), axis=1)
            dists[i] = 99999
            min_dists.append(np.min(dists))

        avg_dist = np.mean(min_dists) if min_dists else 1000.0

        centers_y = [(b[1] + b[3]) / 2.0 for b in valid_boxes]
        heights = [b[3] - b[1] for b in valid_boxes]

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
                row_h = [heights[i] for i in row]
                local_cvs.append(np.std(row_h) / (np.mean(row_h) + 1e-5))

        avg_local_cv = np.mean(local_cvs) if local_cvs else 0.0

        areas = sorted(
            [(b[2] - b[0]) * (b[3] - b[1]) for b in valid_boxes],
            reverse=True
        )

        top_10_idx = max(1, len(areas) // 10)
        bottom_50_idx = max(1, len(areas) // 2)

        top_area_avg = np.mean(areas[:top_10_idx])
        bottom_area_avg = np.mean(areas[-bottom_50_idx:])

        dominance_ratio = top_area_avg / (bottom_area_avg + 1e-5)

        hierarchy_defect = (avg_local_cv * 10.0) + (10.0 / (dominance_ratio + 1.0))

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

        jitter_left = get_jitter([b[0] for b in valid_boxes])
        jitter_right = get_jitter([b[2] for b in valid_boxes])
        jitter_center = get_jitter([(b[0] + b[2]) / 2.0 for b in valid_boxes])

        misalign_score = (jitter_left + jitter_right + jitter_center) / 3.0

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

            return 1.0 / (1.0 + np.exp(-diff_ratio * T))

        p_cov = sigmoid_mapping(cov, cov_thresh, reverse=False)
        p_dist = sigmoid_mapping(dist, dist_thresh, reverse=True)

        score_overcrowd = max(p_cov, p_dist)
        score_misalign = sigmoid_mapping(m_score, m_thresh, reverse=False)
        score_hierarchy = sigmoid_mapping(h_defect, h_thresh, reverse=False)

        return [score_overcrowd, score_misalign, score_hierarchy]


# ==================== 2. 智能校准器 ====================
def normalize_boxes_for_calibration(bboxes):
    """
    保持原始代码逻辑：
    直接使用 tokenizer 后的 bbox。
    padding bbox 会被修正为 1 像素小框，从而保持序列长度口径一致。
    """
    if not bboxes:
        return []

    max_x = max([b[2] for b in bboxes if len(b) == 4] + [1])
    max_y = max([b[3] for b in bboxes if len(b) == 4] + [1])

    scale_x = 1000.0 / max_x if max_x > 1000 else 1.0
    scale_y = 1000.0 / max_y if max_y > 1000 else 1.0

    new_bboxes = []

    for b in bboxes:
        if len(b) != 4:
            continue

        x1 = min(max(0, int(b[0] * scale_x)), 1000)
        y1 = min(max(0, int(b[1] * scale_y)), 1000)
        x2 = min(max(0, int(b[2] * scale_x)), 1000)
        y2 = min(max(0, int(b[3] * scale_y)), 1000)

        if x2 <= x1:
            x2 = min(x1 + 1, 1000)
        if y2 <= y1:
            y2 = min(y1 + 1, 1000)

        new_bboxes.append([x1, y1, x2, y2])

    return new_bboxes


def calibrate_dataset_quantile(all_data):
    print("\n🚀 [智能校准] 正在扫描训练集原始数据...")

    all_covs, all_dists, all_hdefs, all_mscores = [], [], [], []

    for item in all_data:
        # 保持原始训练口径：继续用 item["bbox"] 计算软标签
        bboxes = normalize_boxes_for_calibration(item.get("bbox", []))
        cov, dist, h_defect, m_score = UIHeuristics.calculate_raw_metrics(bboxes)

        all_covs.append(cov)
        all_dists.append(dist)
        all_hdefs.append(h_defect)
        all_mscores.append(m_score)

    GlobalConfig.COVERAGE = float(np.percentile(all_covs, 75))
    GlobalConfig.DIST = float(np.percentile(all_dists, 25))
    GlobalConfig.HIERARCHY_DEFECT = float(np.percentile(all_hdefs, 75))
    GlobalConfig.MISALIGN_SCORE = max(0.1, float(np.percentile(all_mscores, 75)))

    print("🎯 基于训练集的校准完成！参数已固定。")


# ==================== 3. 数据集 ====================
class RealDataset(Dataset):
    def __init__(self, raw_data, is_train=False):
        self.is_train = is_train
        self.raw_data = raw_data

    def _pad(self, seq, val=0):
        if len(seq) > 512:
            return seq[:512]
        return seq + [val] * (512 - len(seq))

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        item = self.raw_data[idx]

        bboxes = item.get("bbox", [])
        input_ids = item.get("input_ids", [])

        # 保持原始口径：用 tokenizer 后的 bbox 计算启发式软标签
        norm_boxes = normalize_boxes_for_calibration(bboxes)
        final_label = UIHeuristics.get_continuous_labels(norm_boxes)

        # 这里也保持原始逻辑：attention_mask 根据 norm_boxes 长度生成
        return {
            "input_ids": torch.tensor(self._pad(input_ids, 0), dtype=torch.long),
            "bbox": torch.tensor(self._pad(norm_boxes, [0, 0, 0, 0]), dtype=torch.long),
            "attention_mask": torch.tensor(self._pad([1] * len(norm_boxes), 0), dtype=torch.long),
            "labels": torch.tensor(final_label).float()
        }


# ==================== 4. 采用自然分布权重 ====================
def calculate_dynamic_pos_weights(train_dataset):
    print("⚖️ 学术模式：采用自然分布权重 (全 1.0)，去除人为倾向")
    return [1.0, 1.0, 1.0]


# ==================== 5. 核心：特征层 + Relation Bias 注入 ====================
class ParallelPerceptionLayer(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()

        self.bbox_encoder = nn.Sequential(
            nn.Linear(8, 128),
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

        self.spatial_encoder = nn.TransformerEncoder(spatial_layer, num_layers=2)

        semantic_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=self.num_heads,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )

        self.semantic_encoder = nn.TransformerEncoder(semantic_layer, num_layers=1)

        self.relation_proj = nn.Linear(11, self.num_heads, bias=False)

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
        cx = (norm_bboxes[:, :, 0] + norm_bboxes[:, :, 2]) / 2.0
        cy = (norm_bboxes[:, :, 1] + norm_bboxes[:, :, 3]) / 2.0
        area = w * h

        dx = cx.unsqueeze(2) - cx.unsqueeze(1)
        dy = cy.unsqueeze(2) - cy.unsqueeze(1)

        w_ratio = torch.log((w.unsqueeze(2) + 1e-5) / (w.unsqueeze(1) + 1e-5))
        h_ratio = torch.log((h.unsqueeze(2) + 1e-5) / (h.unsqueeze(1) + 1e-5))
        area_ratio = torch.log((area.unsqueeze(2) + 1e-5) / (area.unsqueeze(1) + 1e-5))

        left_gap = norm_bboxes[:, :, 0].unsqueeze(2) - norm_bboxes[:, :, 0].unsqueeze(1)
        right_gap = norm_bboxes[:, :, 2].unsqueeze(2) - norm_bboxes[:, :, 2].unsqueeze(1)
        center_gap = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)

        same_row = (torch.abs(dy) < 0.02).float()
        same_col = (torch.abs(dx) < 0.02).float()

        b1 = norm_bboxes.unsqueeze(2)
        b2 = norm_bboxes.unsqueeze(1)

        inter_lt = torch.max(b1[..., :2], b2[..., :2])
        inter_rb = torch.min(b1[..., 2:], b2[..., 2:])
        inter_wh = torch.clamp(inter_rb - inter_lt, min=0)

        inter_area = inter_wh[..., 0] * inter_wh[..., 1]
        iou = inter_area / (area.unsqueeze(2) + area.unsqueeze(1) - inter_area + 1e-6)

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

        rel_bias = self.relation_proj(r_ij)
        rel_bias = rel_bias.permute(0, 3, 1, 2).reshape(B * self.num_heads, N, N)

        return rel_bias

    def forward(self, hidden_states, bboxes, attention_mask):
        norm_bboxes = bboxes.float() / 1000.0

        w = norm_bboxes[:, :, 2] - norm_bboxes[:, :, 0]
        h = norm_bboxes[:, :, 3] - norm_bboxes[:, :, 1]
        cx = (norm_bboxes[:, :, 0] + norm_bboxes[:, :, 2]) / 2.0
        cy = (norm_bboxes[:, :, 1] + norm_bboxes[:, :, 3]) / 2.0

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

        padding_mask = (attention_mask == 0)

        relation_bias = self.compute_relation_bias(norm_bboxes)

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

        spatial_features_masked = spatial_features.masked_fill(~mask, 0.0)
        semantic_features_masked = semantic_features.masked_fill(~mask, 0.0)

        spatial_pool_max = spatial_features.masked_fill(~mask, -1e9).max(dim=1)[0]
        semantic_pool_max = semantic_features.masked_fill(~mask, -1e9).max(dim=1)[0]

        spatial_pool_mean = spatial_features_masked.sum(dim=1) / (mask_float.sum(dim=1) + 1e-9)
        semantic_pool_mean = semantic_features_masked.sum(dim=1) / (mask_float.sum(dim=1) + 1e-9)

        spatial_diff_sq = (spatial_features - spatial_pool_mean.unsqueeze(1)) ** 2
        spatial_pool_std = torch.sqrt(
            (spatial_diff_sq.masked_fill(~mask, 0.0).sum(dim=1)) / (mask_float.sum(dim=1) + 1e-9) + 1e-6
        )

        semantic_diff_sq = (semantic_features - semantic_pool_mean.unsqueeze(1)) ** 2
        semantic_pool_std = torch.sqrt(
            (semantic_diff_sq.masked_fill(~mask, 0.0).sum(dim=1)) / (mask_float.sum(dim=1) + 1e-9) + 1e-6
        )

        spatial_pool = torch.cat([spatial_pool_max, spatial_pool_mean, spatial_pool_std], dim=-1)
        semantic_pool = torch.cat([semantic_pool_max, semantic_pool_mean, semantic_pool_std], dim=-1)

        concat_pool = torch.cat([spatial_pool, semantic_pool], dim=-1)

        weight_overcrowd = F.softmax(self.gate_overcrowd(concat_pool), dim=-1)
        weight_misalign = F.softmax(self.gate_misalign(concat_pool), dim=-1)
        weight_hierarchy = F.softmax(self.gate_hierarchy(concat_pool), dim=-1)

        feat_overcrowd = weight_overcrowd[:, 0:1] * spatial_pool + weight_overcrowd[:, 1:2] * semantic_pool
        feat_misalign = weight_misalign[:, 0:1] * spatial_pool + weight_misalign[:, 1:2] * semantic_pool
        feat_hierarchy = weight_hierarchy[:, 0:1] * spatial_pool + weight_hierarchy[:, 1:2] * semantic_pool

        return feat_overcrowd, feat_misalign, feat_hierarchy


class FinalModel(LayoutLMv3ForTokenClassification):
    def __init__(self, config, dynamic_weights=[1.0, 1.0, 1.0]):
        super().__init__(config)

        hidden_size = config.hidden_size if hasattr(config, "hidden_size") else 768

        self.perception_layer = ParallelPerceptionLayer(hidden_size)

        # 三个独立分类头，对应三类缺陷任务
        self.head_overcrowd = nn.Sequential(
            nn.Linear(hidden_size * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        self.head_misalign = nn.Sequential(
            nn.Linear(hidden_size * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        self.head_hierarchy = nn.Sequential(
            nn.Linear(hidden_size * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        self.loss_fct = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(dynamic_weights),
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

        feat_overcrowd, feat_misalign, feat_hierarchy = self.perception_layer(
            hidden_states=outputs.last_hidden_state,
            bboxes=bbox,
            attention_mask=attention_mask
        )

        logit_o = self.head_overcrowd(feat_overcrowd)
        logit_m = self.head_misalign(feat_misalign)
        logit_h = self.head_hierarchy(feat_hierarchy)

        logits = torch.cat([logit_o, logit_m, logit_h], dim=-1)

        loss = None

        if labels is not None:
            raw_loss = self.loss_fct(logits, labels)

            # 排除 [0.45, 0.55] 区间内的不确定软标签
            confident_mask = ((labels < 0.45) | (labels > 0.55)).float()
            loss = (raw_loss * confident_mask).sum() / (confident_mask.sum() + 1e-9)

        return (loss, logits) if loss is not None else (logits,)


# ==================== 6. 核心报表输出 ====================
class UIMetricsCalculator:
    @staticmethod
    def compute(p: EvalPrediction):
        probs = 1 / (1 + np.exp(-p.predictions))
        continuous_labels = p.label_ids

        true_binary = (continuous_labels > 0.5).astype(int)

        GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"

        print(f"\n{'=' * 70}\n📊 MULTI-LABEL EXPERIMENT REPORT (Macro Avg - Threshold 0.5)\n{'-' * 70}")
        print(f"{'Defect Type':<12} | {'AUC':<8} | {'F1':<8} | {'Prec':<8} | {'Rec':<8}")
        print(f"{'-' * 70}")

        macro_auc = 0
        macro_f1 = 0
        macro_p = 0
        macro_r = 0

        tasks = ["Overcrowd", "Misalign", "Hierarchy"]

        THRESHOLD = 0.5

        for i, task in enumerate(tasks):
            if len(np.unique(true_binary[:, i])) > 1:
                auc = roc_auc_score(true_binary[:, i], probs[:, i])
            else:
                auc = 0.5

            pred_binary = (probs[:, i] > THRESHOLD).astype(int)

            f1 = f1_score(true_binary[:, i], pred_binary, zero_division=0)
            prec = precision_score(true_binary[:, i], pred_binary, zero_division=0)
            rec = recall_score(true_binary[:, i], pred_binary, zero_division=0)

            macro_auc += auc
            macro_f1 += f1
            macro_p += prec
            macro_r += rec

            color = RED if auc < 0.7 else (GREEN if auc > 0.85 else YELLOW)

            print(f"{task:<12} | {color}{auc:.4f}{RESET}   | {f1:.4f}   | {prec:.4f}   | {rec:.4f}")

        print(f"{'-' * 70}")
        print(f"Mean AUC     : {GREEN}{macro_auc / 3:.4f}{RESET}")
        print(f"Macro F1     : {macro_f1 / 3:.4f}")
        print(f"Macro Prec   : {macro_p / 3:.4f}")
        print(f"Macro Rec    : {macro_r / 3:.4f}")
        print(f"{'=' * 70}\n")

        return {
            "macro_auc": macro_auc / 3,
            "macro_f1": macro_f1 / 3
        }


# ==================== 7. Main ====================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    train_path = "/home/zupupdec/LayoutLMV3_Fine_Tuning/split_layoutlmv3_family_merged/train.json"
    val_path = "/home/zupupdec/LayoutLMV3_Fine_Tuning/split_layoutlmv3_family_merged/validation.json"
    test_path = "/home/zupupdec/LayoutLMV3_Fine_Tuning/split_layoutlmv3_family_merged/test.json"

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            print(f"❌ Json not found: {p}")
            return

    train_data = load_json(train_path)
    val_data = load_json(val_path)
    test_data = load_json(test_path)

    calibrate_dataset_quantile(train_data)

    train_set = RealDataset(train_data, is_train=True)
    val_set = RealDataset(val_data, is_train=False)
    test_set = RealDataset(test_data, is_train=False)

    dynamic_weights = calculate_dynamic_pos_weights(train_set)

    print(
        f"📦 启动多标签不互斥识别主实验 | "
        f"训练集: {len(train_data)} | 验证集: {len(val_data)} | 测试集: {len(test_data)}"
    )

    print("⏳ 正在加载模型...")

    config = LayoutLMv3Config.from_pretrained(
        "microsoft/layoutlmv3-base",
        local_files_only=True
    )

    model = FinalModel.from_pretrained(
        "microsoft/layoutlmv3-base",
        config=config,
        dynamic_weights=dynamic_weights,
        local_files_only=True,
        ignore_mismatched_sizes=True
    ).cuda()

    print("🔓 解冻 LayoutLMv3 最后四层 (Layer 8-11)...")

    for name, param in model.layoutlmv3.named_parameters():
        if any(f"layer.{i}" in name for i in range(8, 12)):
            param.requires_grad = True
        else:
            param.requires_grad = False

    for param in model.perception_layer.parameters():
        param.requires_grad = True

    for param in model.head_overcrowd.parameters():
        param.requires_grad = True

    for param in model.head_misalign.parameters():
        param.requires_grad = True

    for param in model.head_hierarchy.parameters():
        param.requires_grad = True

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if "layoutlmv3" in n and p.requires_grad
            ],
            "lr": 2e-5
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if "perception_layer" in n
            ],
            "lr": 1e-4
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if "head_overcrowd" in n
                or "head_misalign" in n
                or "head_hierarchy" in n
            ],
            "lr": 1e-4
        }
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./layoutlmv3_multi_label_main",
            num_train_epochs=30,
            learning_rate=2e-5,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            per_device_train_batch_size=16,
            dataloader_num_workers=4,
            evaluation_strategy="epoch",
            logging_steps=10,
            save_strategy="no",
            fp16=True,
            report_to=None,
            remove_unused_columns=False
        ),
        train_dataset=train_set,
        eval_dataset=val_set,
        optimizers=(optimizer, None),
        compute_metrics=UIMetricsCalculator.compute
    )

    print("🚀 开始跑测...")
    trainer.train()

    print("✅ 训练完毕，正在输出【验证集】最终报告...")
    trainer.evaluate(eval_dataset=val_set)

    print("🎯 正在输出【测试集】最终报告（The Unseen Data）...")
    trainer.evaluate(eval_dataset=test_set)


if __name__ == "__main__":
    main()
