#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from pathlib import Path
from collections import Counter

from transformers import LayoutLMv3TokenizerFast


class Config:
    LABEL_MAPPING = {
        "Text": 0,
        "Button": 1,
        "TextBox": 2,
        "DropList": 3,
        "Date": 4,
        "image": 5,
        "Image": 5
    }

    MODEL_NAME = "microsoft/layoutlmv3-base"
    MAX_LENGTH = 512


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def normalize_bbox(val_dict):
    """
    将 Label Studio 百分比坐标转换为 LayoutLMv3 需要的 [0, 1000] 坐标。
    """
    x = float(val_dict.get("x", 0))
    y = float(val_dict.get("y", 0))
    w = float(val_dict.get("width", 0))
    h = float(val_dict.get("height", 0))

    x1 = int(round(x * 10))
    y1 = int(round(y * 10))
    x2 = int(round((x + w) * 10))
    y2 = int(round((y + h) * 10))

    x1 = max(0, min(x1, 1000))
    y1 = max(0, min(y1, 1000))
    x2 = max(0, min(x2, 1000))
    y2 = max(0, min(y2, 1000))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def extract_elements(item):
    """
    兼容两种 Label Studio 标注格式：

    旧格式：
    rectanglelabels + textarea

    新格式：
    rectangle + labels + textarea
    """
    annotations = item.get("annotations", [])
    if not annotations:
        return []

    results = annotations[0].get("result", [])
    if not results:
        return []

    element_map = {}

    for res in results:
        el_id = res.get("id")
        if not el_id:
            continue

        if el_id not in element_map:
            element_map[el_id] = {
                "bbox": None,
                "label": "Text",
                "text": ""
            }

        el_type = res.get("type")
        val = res.get("value", {})

        if el_type == "rectanglelabels":
            bbox = normalize_bbox(val)
            if bbox is not None:
                element_map[el_id]["bbox"] = bbox

            labels = val.get("rectanglelabels", [])
            if labels:
                element_map[el_id]["label"] = labels[0]

        elif el_type == "rectangle":
            bbox = normalize_bbox(val)
            if bbox is not None:
                element_map[el_id]["bbox"] = bbox

        elif el_type == "labels":
            labels = val.get("labels", [])
            if labels:
                element_map[el_id]["label"] = labels[0]

        elif el_type == "textarea":
            texts = val.get("text", [])
            if texts:
                element_map[el_id]["text"] = str(texts[0])

    elements = []

    for el_id, data in element_map.items():
        if data["bbox"] is None:
            continue

        word = str(data["text"]).strip()
        if not word:
            word = "[UNK]"

        label_name = data["label"]
        label_id = Config.LABEL_MAPPING.get(label_name, 0)

        elements.append({
            "id": el_id,
            "text": word,
            "bbox": data["bbox"],
            "label_name": label_name,
            "label_id": label_id
        })

    # 固定阅读顺序：从上到下、从左到右
    elements.sort(key=lambda e: (e["bbox"][1], e["bbox"][0], e["bbox"][3], e["bbox"][2]))

    return elements


def convert_one_split(split_name, split_dir, out_dir, tokenizer):
    input_json = split_dir / f"{split_name}.json"
    input_meta = split_dir / f"{split_name}_metadata.csv"

    with open(input_json, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        raw_data = [raw_data]

    metadata = read_csv(input_meta)

    if len(raw_data) != len(metadata):
        raise ValueError(
            f"{split_name}: JSON 数量和 metadata 数量不一致："
            f"{len(raw_data)} vs {len(metadata)}"
        )

    final_results = []
    final_metadata = []
    skipped = []

    component_counter = Counter()
    token_label_counter = Counter()

    for idx, (item, meta) in enumerate(zip(raw_data, metadata), start=1):
        task_id = str(item.get("id", ""))
        meta_task_id = str(meta.get("task_id", ""))

        if task_id != meta_task_id:
            raise ValueError(
                f"{split_name}: 第 {idx} 条 task_id 不一致："
                f"json={task_id}, metadata={meta_task_id}"
            )

        elements = extract_elements(item)

        if not elements:
            skipped.append({
                "sample_uid": meta.get("sample_uid", ""),
                "task_id": task_id,
                "reason": "no valid bbox/text"
            })
            continue

        actual_words = [e["text"] for e in elements]
        current_bboxes = [e["bbox"] for e in elements]
        current_labels = [e["label_id"] for e in elements]

        for e in elements:
            component_counter[e["label_name"]] += 1

        encoding = tokenizer(
            actual_words,
            boxes=current_bboxes,
            word_labels=current_labels,
            truncation=True,
            padding="max_length",
            max_length=Config.MAX_LENGTH,
            return_tensors=None
        )

        processed_dict = dict(encoding.data)

        # 修正 padding 位置 label 为 -100
        sequence_labels = []
        for i, mask in enumerate(processed_dict["attention_mask"]):
            if mask == 0:
                sequence_labels.append(-100)
            else:
                sequence_labels.append(int(processed_dict["labels"][i]))

        processed_dict["labels"] = sequence_labels
        processed_dict["bbox"] = [
            [int(coord) for coord in box]
            for box in processed_dict["bbox"]
        ]

        for lab in sequence_labels:
            if lab != -100:
                token_label_counter[lab] += 1

        # 这里保留 metadata，避免后面找不到图片路径和样本来源
        processed_dict["sample_uid"] = meta.get("sample_uid", "")
        processed_dict["dataset"] = meta.get("dataset", "")
        processed_dict["task_id"] = task_id
        processed_dict["image_path"] = meta.get("local_image_path", "")
        processed_dict["filename"] = meta.get("filename", "")
        processed_dict["screen_family_key"] = meta.get("screen_family_key", "")
        processed_dict["screen_family_key_original"] = meta.get("screen_family_key_original", "")
        processed_dict["derived_version"] = meta.get("derived_version", "")

        final_results.append(processed_dict)

        final_metadata.append({
            **meta,
            "num_elements": len(elements),
            "num_words_before_tokenizer": len(actual_words),
            "num_valid_token_labels": sum(1 for x in sequence_labels if x != -100)
        })

        if idx % 500 == 0:
            print(f"{split_name}: 已处理 {idx}/{len(raw_data)}")

    output_json = out_dir / f"{split_name}.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    output_meta = out_dir / f"{split_name}_metadata.csv"
    if final_metadata:
        with open(output_meta, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(final_metadata[0].keys()))
            writer.writeheader()
            writer.writerows(final_metadata)

    skipped_csv = out_dir / f"{split_name}_skipped.csv"
    with open(skipped_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_uid", "task_id", "reason"])
        writer.writeheader()
        writer.writerows(skipped)

    report = {
        "split": split_name,
        "raw_samples": len(raw_data),
        "converted_samples": len(final_results),
        "skipped_samples": len(skipped),
        "component_counter": dict(component_counter),
        "token_label_counter": dict(token_label_counter),
        "outputs": {
            "json": str(output_json),
            "metadata": str(output_meta),
            "skipped": str(skipped_csv)
        }
    }

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-name", default=Config.MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=Config.MAX_LENGTH)
    args = parser.parse_args()

    Config.MODEL_NAME = args.model_name
    Config.MAX_LENGTH = args.max_length

    split_dir = Path(args.split_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(
        Config.MODEL_NAME,
        add_prefix_space=True
    )

    reports = {}

    for split_name in ["train", "validation", "test"]:
        reports[split_name] = convert_one_split(
            split_name=split_name,
            split_dir=split_dir,
            out_dir=out_dir,
            tokenizer=tokenizer
        )

    report_path = out_dir / "convert_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)

    print("\n✅ LayoutLMv3 训练数据转换完成")
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
