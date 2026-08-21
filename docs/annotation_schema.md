


```markdown
# Annotation Schema

## Overview

This document describes the annotation and preprocessing schema used by OwlEye-LD and RA-Layout.

The dataset is designed for mobile UI layout defect detection. Each sample contains token-level text information and bounding-box coordinates for visible UI elements.

## UI Element Representation

Each UI screenshot is converted into a sequence of tokens and bounding boxes.

The core input fields are:

- `input_ids`: token ids after tokenization.
- `bbox`: bounding boxes aligned with the input sequence.
- `attention_mask`: generated during training to indicate valid tokens or UI elements.

Bounding boxes follow the LayoutLM-style coordinate system and are normalized or converted into the range `[0, 1000]`.

## Bounding Box Format

Each bounding box is represented as:

```text
[x1, y1, x2, y2]
