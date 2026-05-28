# Hybrid YOLO + SAM2 Results

Bu not, `hybrid_yolo_sam2_20260528_204958` ciktilarini ozetler.

## Input Files

```text
yolo_raw_conf025.json
yolo_filtered_conf070.json
hybrid_yolo_conf070_sam2.json
visuals_hybrid_conf070_sam2/
```

## Method

```text
test image
  -> YOLO v3-no-erasing raw proposals, conf=0.25
  -> class-aware filtering, default conf=0.70
  -> SAM2 mask refinement on filtered YOLO boxes
```

SAM2 bu deneyde sinif uretmez. Sinif ve confidence YOLO'dan gelir; SAM2 sadece filtrelenmis kutular icin maskeyi yeniden uretir.

## Quantitative Summary

| Stage | Detections | Change |
|---|---:|---:|
| YOLO raw, conf=0.25 | 596 | - |
| Filtered YOLO, conf=0.70 | 219 | -377 |
| YOLO + SAM2 refined | 219 | 0 |

Filtreleme, aday sayisini 596'dan 219'a indirdi. Bu yaklasik %63.3 azalma demektir. SAM2 bekledigi gibi detection sayisini degistirmedi; sadece maskeleri refine etti.

## Class Distribution

### Raw YOLO

| Class | Count |
|---|---:|
| cam | 438 |
| ahsap_dograma | 118 |
| camur_harc | 25 |
| harman_tuglasi | 10 |
| marsilya_tipi_kiremit | 2 |
| cimento_esasli_siva | 2 |
| ahsap_tasiyici | 1 |

### Filtered YOLO / Hybrid Output

| Class | Count |
|---|---:|
| cam | 144 |
| ahsap_dograma | 67 |
| camur_harc | 7 |
| marsilya_tipi_kiremit | 1 |

## Per-Image Counts

| Image | Raw YOLO | Filtered / Hybrid |
|---|---:|---:|
| 1.avif | 66 | 5 |
| 22.jpeg | 128 | 56 |
| 333.jpeg | 13 | 3 |
| 444.jpeg | 23 | 6 |
| 555.avif | 31 | 14 |
| t1.jpg | 179 | 74 |
| t2.jpg | 56 | 19 |
| t3.jpg | 38 | 10 |
| t4.jpg | 62 | 32 |
| t6.jpeg | 0 | 0 |

## Visual Assessment

Filtreleme onceki SAM2 denemesine gore belirgin iyilesme sagladi. Ozellikle bos veya az elemanli gorsellerde cikti daha okunabilir hale geldi. Ancak `22.jpeg`, `t1.jpg` ve `t4.jpg` gibi cephelerde hala cok fazla `cam` ve `ahsap_dograma` bindirmesi var. Bu haliyle hibrit cikti arastirma deneyi olarak degerli, fakat nihai mimari cephe lejant raporu icin hala fazla kalabalik.

Guclu taraflar:

- Raw YOLO gürültüsü ciddi oranda azaldi.
- SAM2, filtrelenmis kutular uzerinden daha duzenli maske poligonlari uretmeye calisti.
- Sinif bilgisi YOLO'dan korundugu icin rapor formatina donusturmek mumkun.

Zayif taraflar:

- `cam` sinifi hala ayni pencere veya cephe bolgesinde fazla tekrarli.
- `ahsap_dograma` ile `cam` arasinda yogun bindirme var.
- SAM2, yanlis YOLO kutusunu duzeltemiyor; sadece verilen kutunun icini segment ediyor.
- `t6.jpeg` icin hic tespit yok; bu gorsel modelin domain/olcek limitini gosteriyor.

## Interpretation

Bu deney, YOLO + SAM2 fikrinin dogru yonde oldugunu ama ana kontrol noktasinin SAM2 degil, YOLO aday filtresi oldugunu gosteriyor.

```text
YOLO proposal kalitesi iyi ise SAM2 maskeyi guzellestiriyor.
YOLO proposal kalitesi zayif ise SAM2 yanlis adayi daha temiz bir maskeye donusturuyor.
```

Bu nedenle siradaki iyilestirme SAM2 fine-tune degil, daha akilli proposal filtering olmalidir.

## Recommended Next Experiment

Bir sonraki deneyde filtreyi daha secici yapalim:

```text
default-conf: 0.78
cam: 0.82
ahsap_dograma: 0.78
camur_harc: 0.72
nms-iou: 0.25
max detections per image/class: cam 25, ahsap_dograma 20
```

Ek olarak kutu alani icin sinif bazli filtre eklenmeli. `cam` ve `ahsap_dograma` icin cok kucuk tekrarlar, `camur_harc` icin de asiri buyuk duvar maskeleri ayrica kontrol edilmeli.

## Paper Note

```text
A hybrid YOLO + SAM2 pipeline was evaluated for architectural facade legend segmentation. YOLO produced class-aware candidate boxes and masks, a rule-based filter removed low-confidence proposals, and SAM2 was used as a zero-shot mask refinement stage. Filtering reduced detections from 596 to 219 across 10 test images. SAM2 preserved the number of detections but refined the mask geometry. The experiment shows that SAM2 is useful as a refinement module, but final report quality depends primarily on the precision of YOLO proposals and the filtering strategy.
```
