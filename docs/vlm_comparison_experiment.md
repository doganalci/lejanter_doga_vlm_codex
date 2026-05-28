# VLM Comparison Experiment

Bu deneyin amaci, VLM'nin cephe lejant raporlamasindaki rolunu iki kosulda karsilastirmaktir.

## Experiment Design

```text
Condition A: image-only VLM
  image -> VLM -> visual analysis + possible report

Condition B: detection-assisted VLM
  image + YOLO/SAM2 JSON -> VLM -> checked facade legend report
```

Condition A, VLM'nin tek basina cephe elemanlarini taniyip taniyamadigini ve segmentasyon tarifi yapip yapamadigini gosterir.

Condition B, YOLO/SAM2 sistem ciktisi verildiginde VLM'nin daha denetlenebilir, sayisal ve teknik bir rapor uretip uretmedigini test eder.

## Expected Outcome

VLM'nin tek basina dogrudan piksel seviyesinde segmentasyon yapmasi beklenmez. Daha gercekci rol:

```text
YOLO: class-aware detection
SAM2: mask refinement
VLM: interpretation, quality critique, and architectural report writing
```

## Evaluation Questions

- VLM image-only kosulda elemanlari dogru adlandiriyor mu?
- VLM image-only kosulda sayi uyduruyor mu, yoksa belirsizlik belirtiyor mu?
- YOLO/SAM2 JSON verilince rapor daha denetlenebilir hale geliyor mu?
- VLM, fazla tekrar ve supheli tespitleri fark edebiliyor mu?
- VLM, segmentasyon yapamadigi durumda bunu acikca soyluyor mu?

## Notebook

```text
notebooks/vlm_comparison_colab.ipynb
```

Notebook uc cikti uretir:

```text
outputs/vlm_comparison/image_only_report.md
outputs/vlm_comparison/detection_assisted_report.md
outputs/vlm_comparison/comparison_report.md
```

## Improved Detection-Assisted Prompt

Ilk VLM denemesinde model toplam tespit sayisini dogru korusa da sinif dagilimini yeniden sayarken hata yapabildi. Bu nedenle gelistirilmis hibrit rapor yaklasiminda VLM'ye ham uzun tespit listesi yerine once dogrulanmis ozet alanlar verilir:

```json
{
  "total_detections": 74,
  "class_counts": {
    "ahsap_dograma": 19,
    "cam": 55
  },
  "sample_detections": []
}
```

Prompt ilkesi:

```text
Use total_detections and class_counts as the only authoritative counts.
Do not recount sample_detections.
```

Bu yapi VLM'nin yorumlama ve raporlama gucunu kullanirken sayisal degerleri deterministik bilgisayarla gorme ciktisindan alir.

## Paper Note

```text
We evaluated the role of a vision-language model in two settings: image-only facade interpretation and detection-assisted report generation. In the image-only setting, the VLM was asked to identify architectural facade elements and produce a legend-style report directly from the photograph. In the detection-assisted setting, the same image was provided together with YOLO/SAM2 outputs. This design tests whether VLMs are more useful as primary segmentation models or as report-generation and quality-control modules over structured computer vision outputs.
```
