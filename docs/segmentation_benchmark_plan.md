# Segmentation Benchmark Plan

Bu benchmark uc yaklasimi karsilastirir:

```text
1. YOLO11 segmentation fine-tune
2. SAM2 prompt-based segmentation
3. VLM / Florence-2 + SAM2 grounded segmentation
```

## 1. YOLO11 Fine-Tuned Segmentation

Bu mevcut ana hattir.

```text
Girdi: image
Cikti: class + confidence + bbox + polygon/mask
Model: yolo11s-seg.pt fine-tuned
Run: outputs/runs/elements-seg-v1
```

Avantajlari:

- Sinif tahmini yapar.
- Maske ve class ayni modelden gelir.
- Roboflow etiketleriyle dogrudan egitilir.
- Hizli inference verir.

Riskleri:

- Veri azsa siniflari karistirir.
- Az ornekli siniflarda zayif kalir.
- Domain disi yeni cephelerde genelleme sinirli olabilir.

## 2. SAM2 Prompt-Based Segmentation

SAM2 sinif bilmez; kendisine prompt gerekir. Bu nedenle iki sekilde test edilir:

### 2.1 YOLO Box + SAM2 Mask

```text
YOLO bbox/class -> SAM2 mask refinement
```

Burada class bilgisi YOLO'dan gelir, maske SAM2 ile iyilestirilmeye calisilir.

Avantaj:

- Maskeler YOLO maskesine gore daha temiz olabilir.
- Sinir detaylari bazi objelerde daha iyi cikabilir.

Risk:

- YOLO bbox yanlissa SAM2 de yanlis bolgeyi segmentler.
- SAM2 malzeme sinifini bilmez.
- Inference maliyeti daha yuksektir.

### 2.2 SAM2 Segment Everything

```text
image -> unlabeled masks
```

Bu mod sinifsiz maske uretir. Karsilastirmada dogrudan class mAP beklenmez; daha cok gorsel kalite ve maske adaylari icin kullanilir.

Avantaj:

- Etiketsiz yeni bolge/malzeme adaylarini yakalayabilir.

Risk:

- Sinif yoktur.
- Cok fazla alakasiz maske uretebilir.
- Lejant raporu icin ek class assignment gerekir.

## 3. VLM / Florence-2 + SAM2 Grounded Segmentation

Bu hatta VLM once gorseldeki nesne/bolge adaylarini bulur veya metin prompt'una gore bbox onerir; SAM2 bu bbox'lari maskeye cevirir.

```text
image + text prompt -> VLM bbox/region proposal -> SAM2 mask
```

Baslangic modeli:

```text
microsoft/Florence-2-base veya microsoft/Florence-2-large
```

Denenecek prompt tipleri:

```text
<OD>
<REGION_PROPOSAL>
<CAPTION_TO_PHRASE_GROUNDING>
```

Avantaj:

- Fine-tune etmeden open-vocabulary benzeri deneme yapilabilir.
- "wooden frame", "brick wall", "glass" gibi daha genel kavramlarla aday bolge bulabilir.
- YOLO'nun hic gormedigi yeni kavramlarda fikir verebilir.

Risk:

- Turkce/yerel mimari sinif adlarini dogrudan iyi anlamayabilir.
- Bbox bulur ama sinir kalitesi icin yine SAM2 gerekir.
- Sinif-malzeme terminolojisinde halusinasyon olabilir.

## 4. Karsilastirma Metrikleri

YOLO ve YOLO+SAM2 icin:

```text
Box Precision
Box Recall
Box mAP50
Box mAP50-95
Mask mAP50
Mask mAP50-95
```

SAM2 segment everything ve VLM+SAM2 icin:

```text
Gorsel kalite
Maske sinir kalitesi
Fazla/eksik maske sayisi
Sinif atanabilirligi
Raporlama icin kullanilabilirlik
Inference suresi
GPU bellek kullanimi
```

## 5. Deney Sirasi

Onerilen sira:

```text
1. YOLO baseline sonucunu sabitle.
2. YOLO bbox + SAM2 mask refinement dene.
3. SAM2 segment everything ile sinifsiz maske kalitesine bak.
4. Florence-2 region proposal / object detection ciktisini SAM2 ile maskeye cevir.
5. Ayni 5 test gorselinde gorsel ve metrik karsilastirma yap.
```

Colab uygulamasi:

```text
4D -> Drive test gorsellerinde YOLO v1/v2/v3 karsilastirmasi
4E -> SAM2 kurulumu ve sam2.1_hiera_tiny checkpoint indirme
4F -> v3 YOLO bbox ciktisini SAM2 mask refinement icin kullanma
9  -> YOLO ve SAM2 ciktilarini Drive reports klasorune kopyalama
```

SAM2 refinement script:

```text
scripts/refine_with_sam2.py
```

Bu script `scripts/infer_yolo.py` ile uretilen JSON icindeki `bbox_xyxy` alanlarini SAM2 box prompt olarak kullanir. Sinif ve confidence YOLO'dan korunur; `sam2_polygons`, `sam2_score` ve SAM2 overlay gorselleri eklenir.

## 6. Beklenti

En guvenilir uretim hatti muhtemelen:

```text
YOLO class detection + SAM2 optional mask refinement + LLM/VLM report
```

VLM'in segmentasyonu tek basina yapmasi beklenmemelidir. VLM daha cok aday bolge bulma, aciklama, raporlama ve eksik/supheli tespitleri yorumlama tarafinda deger katar.

## 7. Kaynaklar

- Meta SAM2: https://ai.meta.com/research/sam2/
- SAM2 GitHub: https://github.com/facebookresearch/sam2
- Ultralytics SAM2 docs: https://docs.ultralytics.com/models/sam-2/
- Florence-2 model card: https://huggingface.co/microsoft/Florence-2-base
