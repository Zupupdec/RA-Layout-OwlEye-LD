#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RA-Layout cross-source multi-label experiment.

"""

import os
import re
import gc
import csv
import json
import random
import hashlib
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import Dataset

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from transformers import (
    LayoutLMv3Config,
    LayoutLMv3ForTokenClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    logging,
)


# ==================== 0. Global settings ====================

warnings.filterwarnings("ignore")
logging.set_verbosity_info()

os.environ["WANDB_DISABLED"] = "true"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.pop("HF_ENDPOINT", None)


# ==================== 1. Paths and config ====================

RICO_PATH = "/home/zupupdec/LayoutLMV3_Fine_Tuning/Rico.json"
OWLEYE_PATH = "/home/zupupdec/LayoutLMV3_Fine_Tuning/OwlEyeL.json"

BASE_MODEL_NAME = "microsoft/layoutlmv3-base"

NUM_EPOCHS = 30
REPORT_EVERY = 3

FIXED_THRESHOLD = 0.5
THRESHOLD_SEARCH_START = 0.05
THRESHOLD_SEARCH_END = 0.95
THRESHOLD_SEARCH_STEP = 0.01

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10

MAX_LENGTH = 512
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
SEED = 42

TARGET_EXP = "all"
# Options: "1", "2", "3", "4", "all"

SIGNATURE_BBOX_GRID = 20
SIGNATURE_MAX_BOXES = 512
SIGNATURE_MAX_TOKENS = 256

RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_ROOT = f"./ra_layout_cross_source_table_results_{RUN_TAG}"

ALL_METRICS_CSV = os.path.join(
    RESULT_ROOT,
    "all_experiments_metrics_summary.csv",
)

TASKS = [
    "Overcrowding",
    "Misalignment",
    "Hierarchy",
]


# ==================== 2. Seed and cleanup ====================

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


# ==================== 3. Screen signature ====================

IMAGE_FIELDS = [
    "image",
    "image_path",
    "img",
    "file",
    "file_name",
    "filename",
    "file_upload",
    "path",
    "url",
]


def stable_hash(obj):
    text = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
    )

    return hashlib.md5(
        text.encode("utf-8"),
    ).hexdigest()


def normalize_image_name(value):
    if value is None:
        return ""

    text = str(value)
    text = text.replace("\\", "/")
    text = text.split("/")[-1]
    text = text.lower().strip()

    text = re.sub(
        r"\.(png|jpg|jpeg|webp|bmp)$",
        "",
        text,
    )

    text = re.sub(
        r"^[0-9a-f]{6,}[-_]",
        "",
        text,
    )

    return text


def get_nested_value(item, fields):
    for key in fields:
        value = item.get(key, None)

        if value is not None and str(value).strip() != "":
            return value

    for parent_key in ["data", "metadata", "meta"]:
        parent = item.get(parent_key, None)

        if not isinstance(parent, dict):
            continue

        for key in fields:
            value = parent.get(key, None)

            if value is not None and str(value).strip() != "":
                return value

    return None


def extract_bboxes_from_item(item):
    boxes = []

    for box in item.get("bbox", []):
        if isinstance(box, dict):
            x1 = float(box.get("x", 0))
            y1 = float(box.get("y", 0))
            x2 = x1 + float(box.get("width", 0))
            y2 = y1 + float(box.get("height", 0))

        elif isinstance(box, (list, tuple)) and len(box) == 4:
            x1, y1, x2, y2 = map(float, box)

        else:
            continue

        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])

    return boxes


def normalize_boxes_for_signature(bboxes):
    if not bboxes:
        return []

    valid = [
        b for b in bboxes
        if len(b) == 4
        and b[2] > b[0]
        and b[3] > b[1]
    ]

    if not valid:
        return []

    max_x = max([b[2] for b in valid] + [1])
    max_y = max([b[3] for b in valid] + [1])

    scale_x = 1000.0 / max_x if max_x > 1000 else 1.0
    scale_y = 1000.0 / max_y if max_y > 1000 else 1.0

    norm_boxes = []

    for box in valid[:SIGNATURE_MAX_BOXES]:
        x1 = min(max(0, box[0] * scale_x), 1000)
        y1 = min(max(0, box[1] * scale_y), 1000)
        x2 = min(max(0, box[2] * scale_x), 1000)
        y2 = min(max(0, box[3] * scale_y), 1000)

        if x2 <= x1 or y2 <= y1:
            continue

        norm_boxes.append([x1, y1, x2, y2])

    return norm_boxes


def quantize_bbox_signature(bboxes, grid=20):
    result = []

    for box in bboxes[:SIGNATURE_MAX_BOXES]:
        x1, y1, x2, y2 = map(float, box)

        if x2 <= x1 or y2 <= y1:
            continue

        q_box = [
            int(round(x1 / grid) * grid),
            int(round(y1 / grid) * grid),
            int(round(x2 / grid) * grid),
            int(round(y2 / grid) * grid),
        ]

        result.append(q_box)

    result = sorted(
        result,
        key=lambda x: (x[1], x[0], x[3], x[2]),
    )

    return result


def get_token_signature(item):
    input_ids = item.get("input_ids", [])

    if isinstance(input_ids, list):
        return input_ids[:SIGNATURE_MAX_TOKENS]

    return []


def build_screen_signature_key(item):
    image_name = normalize_image_name(
        get_nested_value(item, IMAGE_FIELDS),
    )

    raw_boxes = extract_bboxes_from_item(item)
    norm_boxes = normalize_boxes_for_signature(raw_boxes)

    bbox_signature = quantize_bbox_signature(
        norm_boxes,
        grid=SIGNATURE_BBOX_GRID,
    )

    token_signature = get_token_signature(item)

    signature_obj = {
        "image_name": image_name,
        "bbox_signature": bbox_signature,
        "token_signature": token_signature,
    }

    return "screen_signature::" + stable_hash(signature_obj)


def attach_source_and_signature_key(raw_data, source_name):
    new_data = []

    for item in raw_data:
        new_item = dict(item)
        sig_key = build_screen_signature_key(new_item)

        new_item["__source"] = source_name
        new_item["__family_key"] = sig_key
        new_item["screen_signature_key"] = sig_key

        new_data.append(new_item)

    return new_data


def remove_cross_source_signature_overlap(rico_data, owleye_data):
    rico_keys = {
        item["__family_key"]
        for item in rico_data
    }

    owleye_keys = {
        item["__family_key"]
        for item in owleye_data
    }

    overlap_keys = rico_keys & owleye_keys

    rico_clean = [
        item for item in rico_data
        if item["__family_key"] not in overlap_keys
    ]

    owleye_clean = [
        item for item in owleye_data
        if item["__family_key"] not in overlap_keys
    ]

    print("\n" + "=" * 100)
    print("🔒 Cross-source screen_signature_key 过滤报告")
    print("=" * 100)
    print(f"RICO 原始样本数        : {len(rico_data)}")
    print(f"OwlEye 原始样本数      : {len(owleye_data)}")
    print(f"RICO signature 数      : {len(rico_keys)}")
    print(f"OwlEye signature 数    : {len(owleye_keys)}")
    print(f"跨来源重叠 signature 数: {len(overlap_keys)}")
    print(f"RICO 过滤后样本数      : {len(rico_clean)}")
    print(f"OwlEye 过滤后样本数    : {len(owleye_clean)}")
    print("=" * 100)

    return rico_clean, owleye_clean


def group_by_family_key(data):
    groups = {}

    for item in data:
        key = item["__family_key"]

        if key not in groups:
            groups[key] = []

        groups[key].append(item)

    return groups


def split_data_3_ways_by_signature(
    data,
    train_ratio=0.80,
    val_ratio=0.10,
    seed=42,
    source_name="unknown",
):
    groups = group_by_family_key(data)
    keys = list(groups.keys())

    rng = random.Random(seed)
    rng.shuffle(keys)

    total = len(keys)
    train_idx = int(total * train_ratio)
    val_idx = train_idx + int(total * val_ratio)

    train_keys = set(keys[:train_idx])
    val_keys = set(keys[train_idx:val_idx])
    test_keys = set(keys[val_idx:])

    train_data = []
    val_data = []
    test_data = []

    for key in train_keys:
        train_data.extend(groups[key])

    for key in val_keys:
        val_data.extend(groups[key])

    for key in test_keys:
        test_data.extend(groups[key])

    print("\n" + "=" * 100)
    print(f"📦 {source_name} screen_signature_key-level split")
    print("=" * 100)
    print(f"Total samples    : {len(data)}")
    print(f"Total signatures : {total}")

    print(
        f"Train samples    : {len(train_data)} | "
        f"Train signatures: {len(train_keys)}"
    )

    print(
        f"Val samples      : {len(val_data)} | "
        f"Val signatures  : {len(val_keys)}"
    )

    print(
        f"Test samples     : {len(test_data)} | "
        f"Test signatures : {len(test_keys)}"
    )

    print("=" * 100)

    return train_data, val_data, test_data


def assert_no_signature_overlap(name_a, data_a, name_b, data_b):
    keys_a = {
        item["__family_key"]
        for item in data_a
    }

    keys_b = {
        item["__family_key"]
        for item in data_b
    }

    overlap = keys_a & keys_b

    if len(overlap) > 0:
        raise RuntimeError(
            f"❌ screen_signature leakage detected between "
            f"{name_a} and {name_b}: {len(overlap)}"
        )

    print(f"✅ No screen_signature overlap: {name_a} vs {name_b}")


def assert_experiment_signature_clean(
    exp_name,
    train_data,
    val_data,
    test_data,
):
    print("\n" + "-" * 100)
    print(f"🔍 screen_signature_key 严格性检查: {exp_name}")
    print("-" * 100)

    assert_no_signature_overlap(
        "Train",
        train_data,
        "Validation",
        val_data,
    )

    assert_no_signature_overlap(
        "Train",
        train_data,
        "Test",
        test_data,
    )

    assert_no_signature_overlap(
        "Validation",
        val_data,
        "Test",
        test_data,
    )

    print(f"✅ {exp_name} 通过 signature-level leakage 检查")
    print("-" * 100)


# ==================== 4. Label heuristic ====================

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

        valid_boxes = valid_boxes[:MAX_LENGTH]
        count = len(valid_boxes)

        if count < 3:
            return 0.0, 1000.0, 0.0, 0.0

        total_area = sum(
            [
                (b[2] - b[0]) * (b[3] - b[1])
                for b in valid_boxes
            ],
        )

        coverage = total_area / 1000000.0

        centers = np.array(
            [
                [
                    (b[0] + b[2]) / 2.0,
                    (b[1] + b[3]) / 2.0,
                ]
                for b in valid_boxes
            ],
        )

        min_dists = []

        for i in range(count):
            dists = np.sum(
                np.abs(centers - centers[i]),
                axis=1,
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
                    np.std(row_h) / (np.mean(row_h) + 1e-5),
                )

        avg_local_cv = (
            np.mean(local_cvs)
            if local_cvs
            else 0.0
        )

        areas = sorted(
            [
                (b[2] - b[0]) * (b[3] - b[1])
                for b in valid_boxes
            ],
            reverse=True,
        )

        top_idx = max(1, len(areas) // 10)
        bottom_idx = max(1, len(areas) // 2)

        top_area_avg = np.mean(areas[:top_idx])
        bottom_area_avg = np.mean(areas[-bottom_idx:])

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
            groups = [[coords_sorted[0]]]

            for x in coords_sorted[1:]:
                if x - groups[-1][-1] < 20:
                    groups[-1].append(x)
                else:
                    groups.append([x])

            jitter = 0.0

            for group in groups:
                if len(group) >= 3:
                    jitter += np.std(group)

            return jitter

        jitter_left = get_jitter(
            [b[0] for b in valid_boxes],
        )

        jitter_right = get_jitter(
            [b[2] for b in valid_boxes],
        )

        jitter_center = get_jitter(
            [
                (b[0] + b[2]) / 2.0
                for b in valid_boxes
            ],
        )

        misalign_score = (
            jitter_left + jitter_right + jitter_center
        ) / 3.0

        return coverage, avg_dist, hierarchy_defect, misalign_score

    @staticmethod
    def get_continuous_labels(bboxes):
        cov, dist, h_defect, m_score = (
            UIHeuristics.calculate_raw_metrics(bboxes)
        )

        cov_thresh = GlobalConfig.COVERAGE + 1e-5
        dist_thresh = GlobalConfig.DIST + 1e-5
        m_thresh = GlobalConfig.MISALIGN_SCORE + 1e-5
        h_thresh = GlobalConfig.HIERARCHY_DEFECT + 1e-5

        def sigmoid_mapping(val, thresh, reverse=False, temp=10.0):
            if reverse:
                diff_ratio = (thresh - val) / thresh
            else:
                diff_ratio = (val - thresh) / thresh

            diff_ratio = max(
                min(diff_ratio, 5.0),
                -5.0,
            )

            return 1.0 / (
                1.0 + np.exp(-diff_ratio * temp)
            )

        p_cov = sigmoid_mapping(
            cov,
            cov_thresh,
            reverse=False,
        )

        p_dist = sigmoid_mapping(
            dist,
            dist_thresh,
            reverse=True,
        )

        p_m = sigmoid_mapping(
            m_score,
            m_thresh,
            reverse=False,
        )

        p_h = sigmoid_mapping(
            h_defect,
            h_thresh,
            reverse=False,
        )

        return [
            float(max(p_cov, p_dist)),
            float(p_m),
            float(p_h),
        ]


def normalize_boxes_for_calibration(bboxes):
    if not bboxes:
        return []

    valid = [
        b for b in bboxes
        if isinstance(b, (list, tuple))
        and len(b) == 4
        and b[2] > b[0]
        and b[3] > b[1]
    ]

    if not valid:
        return []

    max_x = max([b[2] for b in valid] + [1])
    max_y = max([b[3] for b in valid] + [1])

    scale_x = 1000.0 / max_x if max_x > 1000 else 1.0
    scale_y = 1000.0 / max_y if max_y > 1000 else 1.0

    new_bboxes = []

    for box in valid:
        x1 = min(max(0, int(box[0] * scale_x)), 1000)
        y1 = min(max(0, int(box[1] * scale_y)), 1000)
        x2 = min(max(0, int(box[2] * scale_x)), 1000)
        y2 = min(max(0, int(box[3] * scale_y)), 1000)

        if x2 <= x1:
            x2 = min(x1 + 1, 1000)

        if y2 <= y1:
            y2 = min(y1 + 1, 1000)

        new_bboxes.append([x1, y1, x2, y2])

    return new_bboxes[:MAX_LENGTH]


def calibrate_dataset_quantile(train_data):
    print("\n🚀 [智能校准] 正在扫描当前训练集数据分布...")

    all_covs = []
    all_dists = []
    all_hdefs = []
    all_mscores = []

    for item in train_data:
        raw_boxes = extract_bboxes_from_item(item)

        bboxes = normalize_boxes_for_calibration(
            raw_boxes,
        )

        cov, dist, h_defect, m_score = (
            UIHeuristics.calculate_raw_metrics(bboxes)
        )

        all_covs.append(cov)
        all_dists.append(dist)
        all_hdefs.append(h_defect)
        all_mscores.append(m_score)

    GlobalConfig.COVERAGE = float(
        np.percentile(all_covs, 75),
    )

    GlobalConfig.DIST = float(
        np.percentile(all_dists, 25),
    )

    GlobalConfig.HIERARCHY_DEFECT = float(
        np.percentile(all_hdefs, 75),
    )

    GlobalConfig.MISALIGN_SCORE = max(
        0.1,
        float(np.percentile(all_mscores, 75)),
    )

    print("🎯 基于训练集的校准完成")
    print(f"COVERAGE         = {GlobalConfig.COVERAGE:.6f}")
    print(f"DIST             = {GlobalConfig.DIST:.6f}")
    print(f"HIERARCHY_DEFECT = {GlobalConfig.HIERARCHY_DEFECT:.6f}")
    print(f"MISALIGN_SCORE   = {GlobalConfig.MISALIGN_SCORE:.6f}")


# ==================== 5. Dataset ====================

class RALayoutDataset(Dataset):
    def __init__(self, raw_data):
        self.raw_data = raw_data

    def _pad_ids(self, seq, val=0):
        seq = seq[:MAX_LENGTH]

        return seq + [val] * (
            MAX_LENGTH - len(seq)
        )

    def _pad_boxes(self, seq):
        seq = seq[:MAX_LENGTH]

        return seq + [[0, 0, 0, 0]] * (
            MAX_LENGTH - len(seq)
        )

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        item = self.raw_data[idx]

        raw_boxes = extract_bboxes_from_item(item)

        norm_boxes = normalize_boxes_for_calibration(
            raw_boxes,
        )

        if len(norm_boxes) == 0:
            norm_boxes = [[0, 0, 1, 1]]

        input_ids = item.get("input_ids", [])

        if not isinstance(input_ids, list):
            input_ids = []

        if len(input_ids) == 0:
            input_ids = [0]

        valid_len = min(
            len(input_ids),
            len(norm_boxes),
            MAX_LENGTH,
        )

        input_ids = input_ids[:valid_len]
        model_boxes = norm_boxes[:valid_len]

        label = UIHeuristics.get_continuous_labels(
            norm_boxes,
        )

        return {
            "input_ids": torch.tensor(
                self._pad_ids(input_ids, 0),
                dtype=torch.long,
            ),
            "bbox": torch.tensor(
                self._pad_boxes(model_boxes),
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                self._pad_ids([1] * valid_len, 0),
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                label,
                dtype=torch.float32,
            ),
        }


# ==================== 6. RA-Layout Model ====================

class ParallelPerceptionLayer(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()

        self.bbox_encoder = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        self.num_heads = 8

        spatial_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=self.num_heads,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )

        self.spatial_encoder = nn.TransformerEncoder(
            spatial_layer,
            num_layers=2,
        )

        semantic_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=self.num_heads,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )

        self.semantic_encoder = nn.TransformerEncoder(
            semantic_layer,
            num_layers=1,
        )

        self.relation_proj = nn.Linear(
            11,
            self.num_heads,
            bias=False,
        )

        self.gate_overcrowd = nn.Sequential(
            nn.Linear(hidden_size * 6, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

        self.gate_misalign = nn.Sequential(
            nn.Linear(hidden_size * 6, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

        self.gate_hierarchy = nn.Sequential(
            nn.Linear(hidden_size * 6, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def compute_relation_bias(self, norm_bboxes):
        bsz, seq_len, _ = norm_bboxes.shape

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
            dx ** 2 + dy ** 2 + 1e-6,
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
            b2[..., :2],
        )

        inter_rb = torch.min(
            b1[..., 2:],
            b2[..., 2:],
        )

        inter_wh = torch.clamp(
            inter_rb - inter_lt,
            min=0,
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

        relation = torch.stack(
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
                iou,
            ],
            dim=-1,
        )

        rel_bias = self.relation_proj(relation)

        rel_bias = rel_bias.permute(
            0,
            3,
            1,
            2,
        )

        rel_bias = rel_bias.reshape(
            bsz * self.num_heads,
            seq_len,
            seq_len,
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

        bbox_feat = torch.cat(
            [
                norm_bboxes,
                w.unsqueeze(-1),
                h.unsqueeze(-1),
                cx.unsqueeze(-1),
                cy.unsqueeze(-1),
            ],
            dim=-1,
        )

        padding_mask = attention_mask == 0

        relation_bias = self.compute_relation_bias(
            norm_bboxes,
        )

        spatial_feat = self.spatial_encoder(
            self.bbox_encoder(bbox_feat),
            mask=relation_bias,
            src_key_padding_mask=padding_mask,
        )

        semantic_feat = self.semantic_encoder(
            hidden_states,
            src_key_padding_mask=padding_mask,
        )

        mask = attention_mask.unsqueeze(-1).bool()
        mask_float = attention_mask.unsqueeze(-1).float()

        spatial_masked = spatial_feat.masked_fill(
            ~mask,
            0.0,
        )

        semantic_masked = semantic_feat.masked_fill(
            ~mask,
            0.0,
        )

        spatial_max = spatial_feat.masked_fill(
            ~mask,
            -1e9,
        ).max(dim=1)[0]

        semantic_max = semantic_feat.masked_fill(
            ~mask,
            -1e9,
        ).max(dim=1)[0]

        spatial_mean = spatial_masked.sum(dim=1) / (
            mask_float.sum(dim=1) + 1e-9
        )

        semantic_mean = semantic_masked.sum(dim=1) / (
            mask_float.sum(dim=1) + 1e-9
        )

        spatial_var = (
            spatial_feat - spatial_mean.unsqueeze(1)
        ) ** 2

        semantic_var = (
            semantic_feat - semantic_mean.unsqueeze(1)
        ) ** 2

        spatial_std = torch.sqrt(
            spatial_var.masked_fill(
                ~mask,
                0.0,
            ).sum(dim=1) / (
                mask_float.sum(dim=1) + 1e-9
            ) + 1e-6
        )

        semantic_std = torch.sqrt(
            semantic_var.masked_fill(
                ~mask,
                0.0,
            ).sum(dim=1) / (
                mask_float.sum(dim=1) + 1e-9
            ) + 1e-6
        )

        spatial_pool = torch.cat(
            [
                spatial_max,
                spatial_mean,
                spatial_std,
            ],
            dim=-1,
        )

        semantic_pool = torch.cat(
            [
                semantic_max,
                semantic_mean,
                semantic_std,
            ],
            dim=-1,
        )

        concat_pool = torch.cat(
            [
                spatial_pool,
                semantic_pool,
            ],
            dim=-1,
        )

        weight_o = F.softmax(
            self.gate_overcrowd(concat_pool),
            dim=-1,
        )

        weight_m = F.softmax(
            self.gate_misalign(concat_pool),
            dim=-1,
        )

        weight_h = F.softmax(
            self.gate_hierarchy(concat_pool),
            dim=-1,
        )

        feat_o = (
            weight_o[:, 0:1] * spatial_pool
            + weight_o[:, 1:2] * semantic_pool
        )

        feat_m = (
            weight_m[:, 0:1] * spatial_pool
            + weight_m[:, 1:2] * semantic_pool
        )

        feat_h = (
            weight_h[:, 0:1] * spatial_pool
            + weight_h[:, 1:2] * semantic_pool
        )

        return feat_o, feat_m, feat_h


class RALayoutModel(LayoutLMv3ForTokenClassification):
    def __init__(self, config, dynamic_weights=None):
        super().__init__(config)

        if dynamic_weights is None:
            dynamic_weights = [1.0, 1.0, 1.0]

        hidden_size = config.hidden_size

        self.perception_layer = ParallelPerceptionLayer(
            hidden_size,
        )

        # Three independent classification heads for the three defect tasks.
        head_in_dim = hidden_size * 3

        self.head_overcrowd = nn.Sequential(
            nn.Linear(head_in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        self.head_misalign = nn.Sequential(
            nn.Linear(head_in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        self.head_hierarchy = nn.Sequential(
            nn.Linear(head_in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        self.loss_fct = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                dynamic_weights,
                dtype=torch.float32,
            ),
            reduction="none",
        )

    def forward(
        self,
        input_ids=None,
        bbox=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        if self.loss_fct.pos_weight.device != input_ids.device:
            self.loss_fct.pos_weight = (
                self.loss_fct.pos_weight.to(input_ids.device)
            )

        outputs = self.layoutlmv3(
            input_ids=input_ids,
            bbox=bbox.clamp(0, 1000),
            attention_mask=attention_mask,
        )

        feat_o, feat_m, feat_h = self.perception_layer(
            hidden_states=outputs.last_hidden_state,
            bboxes=bbox,
            attention_mask=attention_mask,
        )

        logit_o = self.head_overcrowd(feat_o)
        logit_m = self.head_misalign(feat_m)
        logit_h = self.head_hierarchy(feat_h)

        logits = torch.cat(
            [
                logit_o,
                logit_m,
                logit_h,
            ],
            dim=-1,
        )

        loss = None

        if labels is not None:
            raw_loss = self.loss_fct(
                logits,
                labels.float(),
            )

            # Exclude soft labels in the uncertain interval [0.45, 0.55].
            confident_mask = (
                (labels < 0.45) | (labels > 0.55)
            ).float()

            loss = (
                raw_loss * confident_mask
            ).sum() / (
                confident_mask.sum() + 1e-9
            )

        if loss is not None:
            return loss, logits

        return logits,

# ==================== 7. Metrics ====================

def sigmoid_np(x):
    return 1.0 / (
        1.0 + np.exp(-np.clip(x, -50, 50))
    )


def unpack_prediction(pred_output):
    logits = pred_output.predictions

    if isinstance(logits, tuple):
        logits = logits[0]

    probs = sigmoid_np(logits)

    labels = pred_output.label_ids

    true_binary = (
        labels > 0.5
    ).astype(int)

    return probs, true_binary


def make_threshold_grid():
    values = np.arange(
        THRESHOLD_SEARCH_START,
        THRESHOLD_SEARCH_END + 1e-9,
        THRESHOLD_SEARCH_STEP,
    )

    return [float(x) for x in values]


def find_best_thresholds_on_validation(pred_output):
    probs, true_binary = unpack_prediction(pred_output)

    grid = make_threshold_grid()
    thresholds = []

    print("\n🔎 Validation-calibrated threshold search")
    print("-" * 80)

    for idx, task in enumerate(TASKS):
        y_true = true_binary[:, idx]
        y_score = probs[:, idx]

        best_th = FIXED_THRESHOLD
        best_f1 = -1.0
        best_p = 0.0
        best_r = 0.0

        for th in grid:
            y_pred = (
                y_score > th
            ).astype(int)

            cur_p = precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )

            cur_r = recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )

            cur_f1 = f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )

            if cur_f1 > best_f1:
                best_f1 = cur_f1
                best_th = th
                best_p = cur_p
                best_r = cur_r

        thresholds.append(best_th)

        print(
            f"{task:<16} | "
            f"best_th={best_th:.2f} | "
            f"val_P={best_p:.3f} | "
            f"val_R={best_r:.3f} | "
            f"val_F1={best_f1:.3f}"
        )

    print("-" * 80)

    return thresholds


def evaluate_prediction(pred_output, thresholds):
    probs, true_binary = unpack_prediction(pred_output)

    result = {
        "per_task": {},
        "macro": {},
        "micro": {},
    }

    macro_auc = 0.0
    macro_auprc = 0.0
    macro_p = 0.0
    macro_r = 0.0
    macro_f1 = 0.0

    micro_true = []
    micro_pred = []
    micro_score = []

    for idx, task in enumerate(TASKS):
        y_true = true_binary[:, idx]
        y_score = probs[:, idx]
        threshold = float(thresholds[idx])

        y_pred = (
            y_score > threshold
        ).astype(int)

        support = int(np.sum(y_true))
        total = int(len(y_true))

        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_score)

            auprc = average_precision_score(
                y_true,
                y_score,
            )
        else:
            auc = 0.5
            auprc = 1.0 if support > 0 else 0.0

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        result["per_task"][task] = {
            "auc": float(auc),
            "auprc": float(auprc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "threshold": float(threshold),
            "support": support,
            "total": total,
            "positive_ratio": (
                float(support / total)
                if total > 0
                else 0.0
            ),
        }

        macro_auc += auc
        macro_auprc += auprc
        macro_p += precision
        macro_r += recall
        macro_f1 += f1

        micro_true.extend(y_true.tolist())
        micro_pred.extend(y_pred.tolist())
        micro_score.extend(y_score.tolist())

    micro_true = np.array(micro_true)
    micro_pred = np.array(micro_pred)
    micro_score = np.array(micro_score)

    if len(np.unique(micro_true)) > 1:
        micro_auc = roc_auc_score(
            micro_true,
            micro_score,
        )

        micro_auprc = average_precision_score(
            micro_true,
            micro_score,
        )
    else:
        micro_auc = 0.5
        micro_auprc = (
            1.0 if np.sum(micro_true) > 0 else 0.0
        )

    micro_p = precision_score(
        micro_true,
        micro_pred,
        zero_division=0,
    )

    micro_r = recall_score(
        micro_true,
        micro_pred,
        zero_division=0,
    )

    micro_f1 = f1_score(
        micro_true,
        micro_pred,
        zero_division=0,
    )

    result["macro"] = {
        "auc": float(macro_auc / 3.0),
        "auprc": float(macro_auprc / 3.0),
        "precision": float(macro_p / 3.0),
        "recall": float(macro_r / 3.0),
        "f1": float(macro_f1 / 3.0),
    }

    result["micro"] = {
        "auc": float(micro_auc),
        "auprc": float(micro_auprc),
        "precision": float(micro_p),
        "recall": float(micro_r),
        "f1": float(micro_f1),
    }

    return result


def print_metrics(metrics, title):
    print("\n" + "=" * 115)
    print(title)
    print("-" * 115)

    print(
        f"{'Task':<16} | "
        f"{'AUC':<7} | "
        f"{'AUPRC':<7} | "
        f"{'P':<7} | "
        f"{'R':<7} | "
        f"{'F1':<7} | "
        f"{'Th':<5} | "
        f"{'Sup':<6} | "
        f"{'PosR':<6}"
    )

    print("-" * 115)

    for task in TASKS:
        item = metrics["per_task"][task]

        print(
            f"{task:<16} | "
            f"{item['auc']:.3f}   | "
            f"{item['auprc']:.3f}   | "
            f"{item['precision']:.3f}   | "
            f"{item['recall']:.3f}   | "
            f"{item['f1']:.3f}   | "
            f"{item['threshold']:.2f} | "
            f"{item['support']:<6} | "
            f"{item['positive_ratio']:.3f}"
        )

    print("-" * 115)

    macro = metrics["macro"]
    micro = metrics["micro"]

    print(
        f"{'Macro Avg':<16} | "
        f"{macro['auc']:.3f}   | "
        f"{macro['auprc']:.3f}   | "
        f"{macro['precision']:.3f}   | "
        f"{macro['recall']:.3f}   | "
        f"{macro['f1']:.3f}"
    )

    print(
        f"{'Micro Avg':<16} | "
        f"{micro['auc']:.3f}   | "
        f"{micro['auprc']:.3f}   | "
        f"{micro['precision']:.3f}   | "
        f"{micro['recall']:.3f}   | "
        f"{micro['f1']:.3f}"
    )

    print("=" * 115)


def append_csv(path, row):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    file_exists = os.path.exists(path)
    fieldnames = list(row.keys())

    with open(
        path,
        "a",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def flatten_metrics(
    exp_name,
    epoch,
    split_name,
    threshold_mode,
    metrics,
):
    row = {
        "run_tag": RUN_TAG,
        "experiment": exp_name,
        "epoch": epoch,
        "split": split_name,
        "threshold_mode": threshold_mode,
        "macro_auc": round(metrics["macro"]["auc"], 3),
        "macro_auprc": round(metrics["macro"]["auprc"], 3),
        "macro_precision": round(
            metrics["macro"]["precision"],
            3,
        ),
        "macro_recall": round(
            metrics["macro"]["recall"],
            3,
        ),
        "macro_f1": round(metrics["macro"]["f1"], 3),
        "micro_auc": round(metrics["micro"]["auc"], 3),
        "micro_auprc": round(metrics["micro"]["auprc"], 3),
        "micro_precision": round(
            metrics["micro"]["precision"],
            3,
        ),
        "micro_recall": round(
            metrics["micro"]["recall"],
            3,
        ),
        "micro_f1": round(metrics["micro"]["f1"], 3),
    }

    for task in TASKS:
        item = metrics["per_task"][task]
        prefix = task.lower()

        row[f"{prefix}_auc"] = round(item["auc"], 3)
        row[f"{prefix}_auprc"] = round(item["auprc"], 3)
        row[f"{prefix}_precision"] = round(
            item["precision"],
            3,
        )
        row[f"{prefix}_recall"] = round(
            item["recall"],
            3,
        )
        row[f"{prefix}_f1"] = round(item["f1"], 3)
        row[f"{prefix}_threshold"] = round(
            item["threshold"],
            3,
        )
        row[f"{prefix}_support"] = item["support"]
        row[f"{prefix}_positive_ratio"] = round(
            item["positive_ratio"],
            3,
        )

    return row


# ==================== 8. Callback ====================

class SaveMetricsEveryNEpochsCallback(TrainerCallback):
    def __init__(
        self,
        exp_name,
        exp_dir,
        val_dataset,
        test_dataset,
        every_n_epochs=3,
    ):
        self.trainer = None
        self.exp_name = exp_name
        self.exp_dir = exp_dir
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(round(state.epoch))

        if epoch <= 0:
            return control

        if epoch % self.every_n_epochs != 0:
            print(
                f"\n📌 RA-Layout | Epoch {epoch} finished. "
                f"Skip saving."
            )
            return control

        print("\n" + "#" * 115)
        print(
            f"📌 RA-Layout | {self.exp_name} | "
            f"Epoch {epoch} | fixed + val-calibrated"
        )
        print("#" * 115)

        fixed_thresholds = [
            FIXED_THRESHOLD,
            FIXED_THRESHOLD,
            FIXED_THRESHOLD,
        ]

        val_pred = self.trainer.predict(
            self.val_dataset,
        )

        best_thresholds = find_best_thresholds_on_validation(
            val_pred,
        )

        val_fixed = evaluate_prediction(
            val_pred,
            fixed_thresholds,
        )

        val_calibrated = evaluate_prediction(
            val_pred,
            best_thresholds,
        )

        test_pred = self.trainer.predict(
            self.test_dataset,
        )

        test_fixed = evaluate_prediction(
            test_pred,
            fixed_thresholds,
        )

        test_calibrated = evaluate_prediction(
            test_pred,
            best_thresholds,
        )

        print_metrics(
            val_fixed,
            title=(
                f"RA-Layout | {self.exp_name} | "
                f"Validation | fixed_0.5 | Epoch {epoch}"
            ),
        )

        print_metrics(
            test_fixed,
            title=(
                f"RA-Layout | {self.exp_name} | "
                f"Test | fixed_0.5 | Epoch {epoch}"
            ),
        )

        print_metrics(
            val_calibrated,
            title=(
                f"RA-Layout | {self.exp_name} | "
                f"Validation | val_calibrated | Epoch {epoch}"
            ),
        )

        print_metrics(
            test_calibrated,
            title=(
                f"RA-Layout | {self.exp_name} | "
                f"Test | val_calibrated | Epoch {epoch}"
            ),
        )

        exp_csv = os.path.join(
            self.exp_dir,
            "metrics_summary.csv",
        )

        rows = [
            flatten_metrics(
                self.exp_name,
                epoch,
                "validation",
                "fixed_0.5",
                val_fixed,
            ),
            flatten_metrics(
                self.exp_name,
                epoch,
                "test",
                "fixed_0.5",
                test_fixed,
            ),
            flatten_metrics(
                self.exp_name,
                epoch,
                "validation",
                "val_calibrated",
                val_calibrated,
            ),
            flatten_metrics(
                self.exp_name,
                epoch,
                "test",
                "val_calibrated",
                test_calibrated,
            ),
        ]

        for row in rows:
            append_csv(exp_csv, row)
            append_csv(ALL_METRICS_CSV, row)

        print(f"✅ CSV 已保存: {exp_csv}")
        print(f"✅ 总表已更新: {ALL_METRICS_CSV}")

        return control


# ==================== 9. Build and train ====================

def calculate_dynamic_weights(train_data):
    print("⚖️ 多标签实验采用自然分布权重：全 1.0")
    return [1.0, 1.0, 1.0]


def build_ra_layout_model(dynamic_weights):
    config = LayoutLMv3Config.from_pretrained(
        BASE_MODEL_NAME,
        local_files_only=True,
    )

    config.num_labels = 3

    model = RALayoutModel.from_pretrained(
        BASE_MODEL_NAME,
        config=config,
        dynamic_weights=dynamic_weights,
        local_files_only=True,
        ignore_mismatched_sizes=True,
    )

    if torch.cuda.is_available():
        model = model.cuda()

    for _, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.layoutlmv3.named_parameters():
        if any(f"layer.{i}" in name for i in range(8, 12)):
            param.requires_grad = True

    for param in model.perception_layer.parameters():
        param.requires_grad = True

    for param in model.head_overcrowd.parameters():
        param.requires_grad = True

    for param in model.head_misalign.parameters():
        param.requires_grad = True

    for param in model.head_hierarchy.parameters():
        param.requires_grad = True

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"\n🔓 RA-Layout 可训练参数量: {trainable:,}")

    return model


def build_optimizer(model):
    groups = []

    layout_params = [
        p for name, p in model.named_parameters()
        if "layoutlmv3" in name and p.requires_grad
    ]

    perception_params = [
        p for name, p in model.named_parameters()
        if "perception_layer" in name and p.requires_grad
    ]

    head_params = [
        p for name, p in model.named_parameters()
        if (
            "head_overcrowd" in name
            or "head_misalign" in name
            or "head_hierarchy" in name
        )
        and p.requires_grad
    ]

    if layout_params:
        groups.append(
            {
                "params": layout_params,
                "lr": 2e-5,
            },
        )

    if perception_params:
        groups.append(
            {
                "params": perception_params,
                "lr": 1e-4,
            },
        )

    if head_params:
        groups.append(
            {
                "params": head_params,
                "lr": 1e-4,
            },
        )

    return torch.optim.AdamW(
        groups,
        weight_decay=0.01,
    )


def make_training_args(**kwargs):
    try:
        return TrainingArguments(**kwargs)

    except TypeError as error:
        if "evaluation_strategy" in str(error):
            kwargs["eval_strategy"] = kwargs.pop(
                "evaluation_strategy",
            )

            return TrainingArguments(**kwargs)

        raise error


def run_experiment(
    exp_name,
    exp_desc,
    train_data,
    val_data,
    test_data,
    dynamic_weights,
):
    cleanup_cuda()

    exp_dir = os.path.join(
        RESULT_ROOT,
        exp_name,
    )

    os.makedirs(
        exp_dir,
        exist_ok=True,
    )

    print("\n" + "=" * 100)
    print(f"🚀 [START] RA-Layout 实验: {exp_name}")
    print(f"📌 说明: {exp_desc}")

    print(
        f"📦 Train: {len(train_data)} | "
        f"Val: {len(val_data)} | "
        f"Test: {len(test_data)}"
    )

    print(
        f"📌 训练 {NUM_EPOCHS} 轮，"
        f"每 {REPORT_EVERY} 轮保存 CSV 指标"
    )

    print("📌 输出 fixed_0.5 + val_calibrated 两套结果")
    print("=" * 100)

    calibrate_dataset_quantile(train_data)

    train_set = RALayoutDataset(train_data)
    val_set = RALayoutDataset(val_data)
    test_set = RALayoutDataset(test_data)

    print("\n⏳ 正在按本地缓存方式加载 RA-Layout...")

    model = build_ra_layout_model(
        dynamic_weights=dynamic_weights,
    )

    optimizer = build_optimizer(model)

    callback = SaveMetricsEveryNEpochsCallback(
        exp_name=exp_name,
        exp_dir=exp_dir,
        val_dataset=val_set,
        test_dataset=test_set,
        every_n_epochs=REPORT_EVERY,
    )

    trainer = Trainer(
        model=model,
        args=make_training_args(
            output_dir=os.path.join(
                exp_dir,
                "trainer_tmp",
            ),
            num_train_epochs=NUM_EPOCHS,
            learning_rate=2e-5,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            dataloader_num_workers=4,
            evaluation_strategy="no",
            logging_steps=10,
            save_strategy="no",
            fp16=torch.cuda.is_available(),
            report_to=[],
            remove_unused_columns=False,
        ),
        train_dataset=train_set,
        eval_dataset=None,
        optimizers=(optimizer, None),
        compute_metrics=None,
        callbacks=[callback],
    )

    callback.trainer = trainer

    print(f"\n🚀 开始训练 RA-Layout: {exp_name}")

    trainer.train()

    print(f"\n✅ RA-Layout 实验完成: {exp_name}")

    del trainer
    del model
    del optimizer

    cleanup_cuda()


# ==================== 10. Main ====================

def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main():
    set_all_seeds(SEED)
    os.makedirs(RESULT_ROOT, exist_ok=True)

    for path in [RICO_PATH, OWLEYE_PATH]:
        if not os.path.exists(path):
            print(f"❌ Json not found: {path}")
            return

    rico_raw = load_json(RICO_PATH)
    owleye_raw = load_json(OWLEYE_PATH)

    print("\n" + "=" * 100)
    print("📌 原始跨来源数据加载完成")
    print("=" * 100)
    print(f"RICO raw samples  : {len(rico_raw)}")
    print(f"OwlEye raw samples: {len(owleye_raw)}")
    print("=" * 100)

    print("\n🔧 正在构造 screen_signature_key...")

    rico_data = attach_source_and_signature_key(
        rico_raw,
        "RICO",
    )

    owleye_data = attach_source_and_signature_key(
        owleye_raw,
        "OwlEye",
    )

    rico_clean, owleye_clean = remove_cross_source_signature_overlap(
        rico_data,
        owleye_data,
    )

    rico_tr, rico_val, rico_te = split_data_3_ways_by_signature(
        rico_clean,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        seed=SEED,
        source_name="RICO",
    )

    owleye_tr, owleye_val, owleye_te = split_data_3_ways_by_signature(
        owleye_clean,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        seed=SEED,
        source_name="OwlEye",
    )

    mixed_train = rico_tr + owleye_tr
    mixed_val = rico_val + owleye_val

    dynamic_weights = calculate_dynamic_weights(mixed_train)

    experiments = [
        {
            "id": "1",
            "name": "Exp1_RICO_to_OwlEye",
            "desc": (
                "RICO train + RICO val -> "
                "OwlEye clean full test"
            ),
            "train": rico_tr,
            "val": rico_val,
            "test": owleye_clean,
        },
        {
            "id": "2",
            "name": "Exp2_OwlEye_to_RICO",
            "desc": (
                "OwlEye train + OwlEye val -> "
                "RICO clean full test"
            ),
            "train": owleye_tr,
            "val": owleye_val,
            "test": rico_clean,
        },
        {
            "id": "3",
            "name": "Exp3_Mixed_to_RICO",
            "desc": (
                "Mixed train + Mixed val -> "
                "RICO held-out test"
            ),
            "train": mixed_train,
            "val": mixed_val,
            "test": rico_te,
        },
        {
            "id": "4",
            "name": "Exp4_Mixed_to_OwlEye",
            "desc": (
                "Mixed train + Mixed val -> "
                "OwlEye held-out test"
            ),
            "train": mixed_train,
            "val": mixed_val,
            "test": owleye_te,
        },
    ]

    if TARGET_EXP.lower() == "all":
        run_list = experiments
        print("🚦 当前模式: 连续运行全部 4 个实验")
    else:
        run_list = [
            exp for exp in experiments
            if exp["id"] == TARGET_EXP
        ]

        if not run_list:
            print("❌ TARGET_EXP 必须是 '1'/'2'/'3'/'4'/'all'")
            return

        print(f"🚦 当前模式: 单独运行实验 {TARGET_EXP}")

    print("\n" + "=" * 100)
    print("🏆 数据准备完成")
    print("只保存 CSV，不保存 checkpoint，不保存 json")
    print("已启用 screen_signature 过滤 + signature-level split")
    print("将输出 fixed_0.5 + val_calibrated 两套结果")
    print(f"即将运行实验数量: {len(run_list)}")
    print(f"总 CSV: {ALL_METRICS_CSV}")
    print("=" * 100)

    for exp in run_list:
        print(f"\n👉 准备阶段: {exp['desc']}")

        assert_experiment_signature_clean(
            exp_name=exp["name"],
            train_data=exp["train"],
            val_data=exp["val"],
            test_data=exp["test"],
        )

        run_experiment(
            exp_name=exp["name"],
            exp_desc=exp["desc"],
            train_data=exp["train"],
            val_data=exp["val"],
            test_data=exp["test"],
            dynamic_weights=dynamic_weights,
        )

    print("\n🎉 所有 RA-Layout 跨来源实验已完成")
    print(f"📁 结果目录: {RESULT_ROOT}")
    print(f"📄 汇总 CSV: {ALL_METRICS_CSV}")


if __name__ == "__main__":
    main()
