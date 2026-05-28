# SAM2 Refinement Results

Bu not, Drive test gorsellerinde YOLO v3-no-erasing ciktisinin SAM2 ile mask refinement denemesini ozetler.

## Input Files

```text
drive_test_v3-no-erasing.json
sam2_refined_v3.json
visuals_v3/
```

## Method

```text
image
  -> YOLO v3-no-erasing class + confidence + bbox
  -> SAM2 box prompt
  -> SAM2 refined mask polygon
```

SAM2 sinif tahmini yapmaz. Bu deneyde sinif ve confidence degerleri YOLO'dan korunmus, sadece bbox icindeki maske SAM2 ile yeniden uretilmistir.

## Quantitative Summary

```text
Test images: 10
YOLO detections: 596
SAM2 refined detections: 596
Mean YOLO confidence: 0.613
Confidence range: 0.251 - 0.950
```

Class distribution:

| Class | Count |
|---|---:|
| cam | 438 |
| ahsap_dograma | 118 |
| camur_harc | 25 |
| harman_tuglasi | 10 |
| marsilya_tipi_kiremit | 2 |
| cimento_esasli_siva | 2 |
| ahsap_tasiyici | 1 |

Per-image detection density:

| Image | Detections | conf >= 0.5 | conf >= 0.7 |
|---|---:|---:|---:|
| 1.avif | 66 | 29 | 9 |
| 22.jpeg | 128 | 91 | 60 |
| 333.jpeg | 13 | 5 | 3 |
| 444.jpeg | 23 | 12 | 6 |
| 555.avif | 31 | 23 | 14 |
| t1.jpg | 179 | 119 | 81 |
| t2.jpg | 56 | 37 | 21 |
| t3.jpg | 38 | 23 | 11 |
| t4.jpg | 62 | 49 | 38 |
| t6.jpeg | 0 | 0 | 0 |

SAM2 mask area summary:

```text
Min area: 7 px
Median area: 117 px
Mean area: 1224 px
Max area: 51469 px
```

## Visual Assessment

The SAM2 overlays show many overlapping labels and masks. This is mostly caused by the YOLO v3-no-erasing model producing many candidate boxes. SAM2 then refines every candidate, so it does not reduce the number of false positives.

Observed issues:

- Too many `cam` detections on the same facade.
- Heavy overlap between `cam` and `ahsap_dograma` boxes.
- Large material masks such as `camur_harc` sometimes cover broad wall regions.
- Some masks are visually plausible, but the scene-level output is too noisy for a clean architectural legend report.
- Image `t6.jpeg` produced no detections, indicating possible domain/scale mismatch.

## Interpretation

SAM2 zero-shot refinement is not the primary bottleneck. The main bottleneck is YOLO proposal quality:

```text
If YOLO bbox is noisy -> SAM2 refines noisy prompts.
If YOLO bbox is correct -> SAM2 can produce a cleaner local mask.
```

Therefore SAM2 should not be fine-tuned yet. The next experiment should first reduce YOLO false positives with threshold tuning and possibly class-specific filtering.

## Next Experiment

Run YOLO v3-no-erasing on the same Drive test images with stricter confidence thresholds:

```text
conf=0.50
conf=0.60
conf=0.70
```

Recommended next candidate:

```text
conf=0.70
```

Reason: in the current output, detections drop substantially at `conf >= 0.7`, while many high-confidence window/dograma detections remain.

After threshold tuning, rerun SAM2 only on the selected threshold output.

## Paper Note

This can be reported as:

```text
SAM2 was tested as a zero-shot mask refinement stage using YOLO bounding boxes as prompts. The refinement preserved YOLO class assignments but did not reduce false positives. Visual results showed that SAM2 mask quality depends strongly on the quality of the YOLO proposals. Therefore, YOLO threshold tuning and proposal filtering are required before SAM2 fine-tuning is meaningful.
```
