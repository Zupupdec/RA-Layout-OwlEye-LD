
---

## 3. `docs/label_mapping.md`



```markdown
# Label Mapping

## Layout Defect Labels

RA-Layout uses three layout defect labels in the main multi-label experiment.

| Index | Label Name | Description |
|---|---|---|
| 0 | Overcrowding | Dense UI elements, insufficient spacing, or excessive local stacking. |
| 1 | Misalignment | Obvious deviation from expected alignment lines or layout structure. |
| 2 | Poor Visual Hierarchy | Weak visual priority, unclear information structure, or poor emphasis. |

The label vector is ordered as:

```text
[Overcrowding, Misalignment, Poor Visual Hierarchy]
