# RA-Layout-OwlEye-LD

This repository provides the experiment code and configuration files for **RA-Layout**, a LayoutLMv3-based model for mobile UI layout defect detection on OwlEye-LD.

RA-Layout is designed for detecting three layout defect categories:

- Overcrowding
- Misalignment
- Poor Visual Hierarchy

The main task is formulated as a multi-label classification problem, since a UI screenshot may contain zero, one, or multiple layout defects.

## Repository Structure

```text
RA-Layout-OwlEye-LD/
├── configs/
│   ├── main.yaml
│   ├── binary.yaml
│   ├── ablation.yaml
│   └── cross_source.yaml
├── data/
│   ├── sample/
│   ├── splits/
│   └── metadata/
├── scripts/
│   ├── train_main.py
│   ├── train_binary.py
│   ├── run_ablation.py
│   └── run_cross_source.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Model Overview

RA-Layout uses `microsoft/layoutlmv3-base` as the backbone and adds a Parallel Perception Layer (PPL) to model spatial and semantic layout information.

The model includes:

- LayoutLMv3 backbone
- Parallel Perception Layer
- relation-aware attention bias
- handcrafted geometric features
- masked statistical pooling
- task-specific gating
- independent classification heads for the three defect categories

The three output heads correspond to:

```text
[Overcrowding, Misalignment, Poor Visual Hierarchy]
```

## Main Multi-label Experiment

The main experiment detects the three layout defect types simultaneously.

Run:

```bash
python scripts/train_main.py
```

Main settings:

- backbone: `microsoft/layoutlmv3-base`
- training epochs: 30
- batch size: 16
- threshold for metric computation: 0.5
- loss function: `BCEWithLogitsLoss`
- confidence mask: excludes soft labels in `[0.45, 0.55]`

The confidence mask follows:

```python
confident_mask = ((labels < 0.45) | (labels > 0.55)).float()
```

This means labels in the uncertain interval `0.45 <= label <= 0.55` are excluded from the loss.

## Binary Auxiliary Experiment

The binary experiment detects whether a screenshot is defective or normal.

Run:

```bash
python scripts/train_binary.py
```

The binary label is derived from the three defect scores:

```text
Defective = 1 if any defect score is >= 0.5.
Normal = 0 otherwise.
```

The binary experiment uses a single binary classification head and does not use the confidence mask, because the labels are already hard binary labels.

## Ablation Study

Run:

```bash
python scripts/run_ablation.py
```

The ablation study includes:

| Ablation | Description |
|---|---|
| `wo_relation` | Remove relation-aware attention bias while retaining the remaining PPL structure. |
| `wo_geometry` | Remove handcrafted geometric features and use only original bounding-box coordinates. |
| `wo_ocr` | Remove OCR text token information. |
| `wo_conf_mask` | Remove the confidence mask and compute BCE loss on all soft labels. |
| `wo_pseudo_weak` | Binarize training soft labels into hard 0/1 labels. |
| `wo_ppl_rab` | Remove PPL as a whole. |

## Cross-source Experiment

Run:

```bash
python scripts/run_cross_source.py
```

The cross-source experiment evaluates source-level generalization between RICO-derived and OwlEye-derived subsets.

It reports two threshold settings:

- fixed threshold: `0.5`
- validation-calibrated threshold: thresholds are selected on the validation set by maximum F1 and then applied to the test set

The test set is never used to search thresholds.

## Data

The full OwlEye-LD dataset is not included in this repository due to copyright and privacy constraints.

This repository only provides:

- sample data format
- split information
- metadata descriptions
- experiment scripts
- configuration files

For local reproduction, place the processed main dataset under:

```text
data/processed/owleye_ld/train.json
data/processed/owleye_ld/validation.json
data/processed/owleye_ld/test.json
```

For cross-source experiments, place source-specific data under:

```text
data/processed/source_subsets/rico.json
data/processed/source_subsets/owleye.json
```

The `data/processed/` directory is ignored by Git and should not be uploaded.

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
```

Required fields:

- `input_ids`: token ids after preprocessing
- `bbox`: bounding boxes aligned with tokens or visible UI elements

Optional but recommended fields:

- `id`: sample identifier
- `source`: data source, such as `rico` or `owleye`

## Configuration Files

The `configs/` directory records the experimental settings used in the paper.

| File | Description |
|---|---|
| `configs/main.yaml` | Main multi-label experiment settings. |
| `configs/binary.yaml` | Binary auxiliary experiment settings. |
| `configs/ablation.yaml` | Ablation experiment settings. |
| `configs/cross_source.yaml` | Cross-source validation settings. |

The current scripts mainly use internal constants. If needed, update the path variables in the scripts to match your local data placement.

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

The main dependencies include:

- PyTorch
- Transformers
- NumPy
- scikit-learn
- pandas
- Pillow
- tqdm
- PyYAML

The scripts use `microsoft/layoutlmv3-base`. If running in offline mode, make sure the model files have already been cached locally.

## Reproduction

After placing the processed data locally, run the experiments with:

```bash
python scripts/train_main.py
python scripts/train_binary.py
python scripts/run_ablation.py
python scripts/run_cross_source.py
```

No checkpoints, trained model weights, or full dataset files are included in this repository.

## Notes

- The full dataset is not publicly released in this repository.
- The repository is intended to provide code, configuration files, and reproducibility information.
- Training labels are generated by the heuristic label construction procedure in the scripts.
- Continuous labels are binarized with a threshold of 0.5 for metric computation.
