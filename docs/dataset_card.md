# Dataset Card

## Dataset Name

OwlEye-LD

## Overview

OwlEye-LD is a mobile UI screenshot dataset for layout defect detection. It is used to evaluate RA-Layout on three layout defect categories:

- Overcrowding
- Misalignment
- Poor Visual Hierarchy

The dataset contains 9,992 valid mobile UI screenshots after cleaning, annotation review, and near-duplicate handling.

## Task Setting

The main task is multi-label layout defect detection. Each screenshot may contain zero, one, or multiple layout defects.

The three defect labels are:

| Label ID | Defect Type |
|---|---|
| 0 | Overcrowding |
| 1 | Misalignment |
| 2 | Poor Visual Hierarchy |

A screenshot without any of the three defects is treated as Normal.

## Data Format

Each processed sample should contain at least:

```json
{
  "id": "sample_0001",
  "source": "owleye",
  "input_ids": [0, 101, 2023, 2003, 102],
  "bbox": [
    [0, 0, 1000, 1000],
    [120, 80, 320, 130],
    [120, 160, 600, 220]
  ]
}
