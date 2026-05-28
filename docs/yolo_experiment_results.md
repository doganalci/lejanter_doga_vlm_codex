# YOLO Segmentation Experiment Results

Bu dokuman Colab uzerinde kosulan uc YOLO11s segmentation deneyinin makale/rapor icin saklanacak ozetidir.

Model dosyalari GitHub'a eklenmemistir. Tum agirliklar ve egitim ciktisi asagidaki zip icinde saklanmistir:

```text
/Users/doganalci/Downloads/gtu_yolo_models_and_results.zip
```

Ayrica kalici Colab kullanimi icin Google Drive klasor yapisi kullanilmaktadir:

```text
Google Drive shared folder
  dataset/  -> GTU_MIMARI_LEJANT.yolov11.zip
  weights/  -> elements-seg-v1/v2/v3 best.pt dosyalari
  test/     -> dis test gorselleri
  reports/  -> egitim metrikleri, configler, epoch sonuclari ve test gorselleri
```

Paylasilan klasor:

```text
https://drive.google.com/drive/folders/1QJD49ylJs9PpidPDQEcCMhx0RltWUfTT
```

Zip icerigi:

```text
outputs/runs/elements-seg-v1/
outputs/runs/elements-seg-v2-aug-controlled/
outputs/runs/elements-seg-v3-no-erasing/
```

Her run icinde:

```text
weights/best.pt
weights/last.pt
results.csv
results.png
confusion_matrix.png
confusion_matrix_normalized.png
val_batch0_labels.jpg
val_batch0_pred.jpg
BoxPR_curve.png
MaskPR_curve.png
BoxF1_curve.png
MaskF1_curve.png
```

## Dataset

```text
Task: Instance Segmentation
Format: YOLOv11
Total images: 52
Train: 42
Validation: 5
Test: 5
Validation instances: 153
Classes: 17
```

Validation set cok kucuk oldugu icin metrikler deneyler arasinda oynak olabilir. Sonuclar mutlaka `val_batch0_pred.jpg`, confusion matrix ve test/custom inference gorselleriyle birlikte yorumlanmalidir.

## Experiment Summary

### v1: Default / Baseline

```text
Run: elements-seg-v1
Model: yolo11s-seg.pt
Epochs completed: 172
Best epoch: 72
Early stopping: no improvement in last 100 epochs
Training time: 0.050 h
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
```

Best model validation:

```text
Box Precision:      0.384
Box Recall:         0.255
Box mAP50:          0.326
Box mAP50-95:       0.188
Mask Precision:     0.358
Mask Recall:        0.230
Mask mAP50:         0.294
Mask mAP50-95:      0.0774
```

Notlar:

- Genel olarak en dengeli baseline.
- Precision ve recall dusuk, ancak v2'ye gore genel mAP daha iyi.
- `cam` sinifi makul; zayif siniflar `ahsap_tasiyici`, `alaturka_kiremit`, `demir`, `toprak_siva`.

### v2: Controlled Augmentation

```text
Run: elements-seg-v2-aug-controlled
Model: yolo11s-seg.pt
Epochs completed: 471
Best epoch: 271
Early stopping: no improvement in last 200 epochs
Training time: 0.119 h
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
```

Controlled augmentation settings:

```text
mosaic=0.3
erasing=0.0
scale=0.3
translate=0.05
hsv_h=0.01
hsv_s=0.4
hsv_v=0.3
fliplr=0.5
flipud=0.0
perspective=0.0
```

Best model validation:

```text
Box Precision:      0.593
Box Recall:         0.191
Box mAP50:          0.280
Box mAP50-95:       0.145
Mask Precision:     0.519
Mask Recall:        0.195
Mask mAP50:         0.258
Mask mAP50-95:      0.0678
```

Notlar:

- Precision v1'e gore artti, recall ve genel mAP dustu.
- Model daha az ama daha emin tahmin yapiyor.
- `cam` sinifinda belirgin iyilesme var:
  - v1 cam Mask mAP50: 0.354
  - v2 cam Mask mAP50: 0.461
  - v1 cam Mask mAP50-95: 0.164
  - v2 cam Mask mAP50-95: 0.263
- Genel model olarak v1'in gerisinde, ancak sinif bazli analiz icin saklanmali.

### v3: No Erasing

```text
Run: elements-seg-v3-no-erasing
Model: yolo11s-seg.pt
Epochs completed: 300
Training time: 0.085 h
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
```

Main change:

```text
erasing=0.0
```

Best model validation:

```text
Box Precision:      0.152
Box Recall:         0.395
Box mAP50:          0.329
Box mAP50-95:       0.145
Mask Precision:     0.152
Mask Recall:        0.380
Mask mAP50:         0.309
Mask mAP50-95:      0.100
```

Notlar:

- En iyi genel `Mask mAP50` ve `Mask mAP50-95` degerini verdi.
- Recall en yuksek model oldu.
- Precision ciddi dustu; daha fazla false positive uretme riski var.
- `cam` sinifi en iyi sonucu v3'te verdi:
  - Mask mAP50: 0.526
  - Mask mAP50-95: 0.311

## Comparison Table

| Run | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v1 default | 0.384 | 0.255 | 0.326 | 0.188 | 0.358 | 0.230 | 0.294 | 0.0774 | Balanced baseline |
| v2 controlled aug | 0.593 | 0.191 | 0.280 | 0.145 | 0.519 | 0.195 | 0.258 | 0.0678 | Higher precision, lower recall |
| v3 no erasing | 0.152 | 0.395 | 0.329 | 0.145 | 0.152 | 0.380 | 0.309 | 0.100 | Highest recall and mask mAP, more false positives |

## Current Model Choice

Recommended working model:

```text
elements-seg-v3-no-erasing/weights/best.pt
```

Reason:

- Highest `Mask mAP50`: 0.309
- Highest `Mask mAP50-95`: 0.100
- Highest mask recall: 0.380
- Better for exploratory segmentation where missing facade elements is more costly than extra candidates.

Secondary comparison model:

```text
elements-seg-v1/weights/best.pt
```

Reason:

- More balanced than v3.
- Better choice when false positives need to be lower.

## Next Evaluation Step

Run inference with v3 at multiple confidence thresholds:

```text
conf=0.25
conf=0.40
conf=0.50
```

Goal:

- Keep v3's high recall.
- Reduce false positives by threshold tuning.
- Compare overlay outputs visually.

## Research Notes

For the article/report:

- Controlled augmentation did not improve global metrics, but improved `cam` class.
- Disabling erasing improved mask recall and mask mAP.
- The dominant limitation is still dataset size and class imbalance, not model capacity.
- Validation has only 5 images; results should be described as preliminary baseline, not final generalization performance.
