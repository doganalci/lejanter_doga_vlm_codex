# GTU Mimari Lejant Modelleme Teknik Raporu

## 1. Proje Amaci

Bu calismanin amaci, tarihi/yoresel binalarin dis cephelerinde gorulen mimari malzeme ve lejant elemanlarini goruntu uzerinden otomatik olarak tespit etmek, segmentlemek, saymak ve raporlanabilir bir yapiya donusturmektir.

Hedef ciktilar uc katmanlidir:

1. Goruntu uzerinde eleman/malzeme segmentasyonu.
2. Her tespit icin sinif, guven skoru, bounding box ve polygon/mask ciktisi.
3. Bu yapisal ciktilardan mimari cephe lejant raporu uretimi.

Bu nedenle ilk asamada dogrudan VLM fine-tuning yapmak yerine, olculebilir ve denetlenebilir bir instance segmentation modeli kurulmustur. VLM/LLM katmani daha sonra detector/segmenter ciktisini teknik rapora cevirmek icin kullanilacaktir.

## 2. Veri Seti

Veri Roboflow uzerinde etiketlenmis instance segmentation dataset'inden alinmistir.

Export bilgisi:

```text
Format: YOLOv11
Task: Instance Segmentation
Kaynak dosya: GTU_MIMARI_LEJANT.yolov11.zip
Yerel klasor: data/roboflow/gtu-mimari-lejant
```

Roboflow export paketinde yalnizca `train` split'i geldigi icin repo icinde `scripts/split_yolo_dataset.py` ile lokal train/valid/test ayrimi yapilmistir.

Son split:

```text
Toplam goruntu: 52
Train: 42 image / 42 label
Valid: 5 image / 5 label
Test: 5 image / 5 label
Sinif sayisi: 17
```

Siniflar:

```text
0  ahsap_dograma
1  ahsap_kaplamali_ahsap_karkas_duvar
2  ahsap_tasiyici
3  alaturka_kiremit
4  cam
5  camur_harc
6  cimento_esasli_siva
7  delikli_tugla
8  demir
9  dogal_tas_duvar_harcli
10 harman_tuglasi
11 kirec_badana
12 kirec_esasli_siva
13 kirec_harc
14 marsilya_tipi_kiremit
15 toprak_siva
16 tugla_duvar
```

Kullanilan dataset config dosyasi:

```text
configs/elements_dataset.yaml
```

## 3. Kullanilan Yontem

### 3.1 Neden YOLO Segmentation?

Bu problemde ihtiyac yalnizca "goruntude hangi sinif var?" sorusu degildir. Mimari cephe elemanlarinin konumu, siniri ve sayisi gerekir. Bu nedenle classification yerine instance segmentation secilmistir.

YOLO segmentation secilme nedenleri:

- Roboflow YOLOv11 export formatini dogrudan okuyabilir.
- Her obje icin sinif + confidence + bounding box + polygon/mask ciktisi verir.
- Kucuk veri setlerinde hizli baseline alinabilir.
- Colab GPU uzerinde kolayca egitilebilir.
- Sonuclar `JSON`, gorsel overlay ve raporlama pipeline'ina rahat aktarilir.

### 3.2 Neden Ilk Asamada VLM Fine-Tuning Yapilmadi?

VLM fine-tuning daha pahali, daha karmasik ve dogrulugu daha zor olculen bir yoldur. Bu projede ilk hedef sayilabilir, denetlenebilir ve geometrik olarak kontrol edilebilir tespitler uretmektir.

Bu nedenle mimari:

```text
Roboflow labels
  -> YOLO instance segmentation
  -> JSON / overlay outputs
  -> cephe/eleman sayimlari
  -> LLM veya VLM ile teknik rapor
```

seklinde kurulmustur.

VLM/LLM katmani modelin yerine gecmez; model ciktisini mimari dile ceviren raporlama katmani olarak konumlandirilir.

## 4. Repo Icerisinde Yapilanlar

### 4.1 Proje Iskeleti

Olusturulan ana klasorler:

```text
configs/
docs/
notebooks/
scripts/
data/roboflow/
outputs/
```

`data/` ve `outputs/` altindaki buyuk veri/model dosyalari GitHub'a alinmamistir. Bu dosyalar `.gitignore` ile disarida tutulur.

### 4.2 Dataset Config

`configs/elements_dataset.yaml` dosyasi gercek Roboflow siniflari ile hazirlanmistir.

Bu dosya Ultralytics YOLO egitiminin hangi klasorlerden veri okuyacagini ve sinif id'lerinin hangi isimlere karsilik geldigini tanimlar.

### 4.3 Split Scripti

Eklenen script:

```text
scripts/split_yolo_dataset.py
```

Gorevi:

- Sadece `train/images` ve `train/labels` iceren Roboflow export'unu okur.
- Belirlenen oranlarda `valid` ve `test` klasorleri olusturur.
- Image/label eslesmelerini korur.
- Varsayilan oranlarla 52 goruntuyu 42/5/5 olarak ayirmistir.

Calistirilan komut:

```bash
python scripts/split_yolo_dataset.py \
  --root data/roboflow/gtu-mimari-lejant \
  --valid 0.1 \
  --test 0.1
```

### 4.4 Egitim Scripti

Eklenen script:

```text
scripts/train_yolo.py
```

Gorevi:

- Ultralytics YOLO modelini yukler.
- Detection veya segmentation task'i ile egitim baslatir.
- Run ciktilarini `outputs/runs/` altina yazar.

Ilk baseline icin secilen model:

```text
yolo11s-seg.pt
```

Bu secim, hiz ve kapasite arasinda dengeli oldugu icin yapilmistir. T4 GPU'da bellek sorunu olursa `yolo11n-seg.pt` ve daha dusuk `imgsz` denenebilir.

### 4.5 Inference Scripti

Eklenen script:

```text
scripts/infer_yolo.py
```

Gorevi:

- Egitilmis `best.pt` agirligini yukler.
- Test goruntulerinde tahmin calistirir.
- Her goruntu icin sinif, confidence, bbox ve polygon ciktisini JSON'a yazar.
- Istenirse gorsel overlay kaydeder.

### 4.6 Raporlama Scriptleri

Eklenen scriptler:

```text
scripts/report_from_detections.py
scripts/report_with_vllm.py
```

`report_from_detections.py`:

- Detection/segmentation JSON ciktisini okur.
- Her siniftan kac adet tespit edildigini raporlar.
- Cephe maskesi eklendiginde elemanlari ilgili cepheye atayabilir.

`report_with_vllm.py`:

- JSON ciktisini OpenAI-compatible vLLM endpoint'ine gonderir.
- Turkce mimari cephe lejant raporu uretir.
- Prompt icinde modelin sayi uydurmamasi ve sadece JSON verisini kullanmasi istenir.

### 4.7 Colab Notebook

Eklenen notebook:

```text
notebooks/train_colab.ipynb
```

Notebook akisi:

1. GPU kontrolu.
2. GitHub repo clone.
3. Python dependency kurulumu.
4. Roboflow zip upload veya Google Drive'dan zip kopyalama.
5. Dataset unzip ve split kontrolu.
6. YOLO segmentation egitimi.
7. Opsiyonel kontrollu augmentation deneyi.
8. Test set inference.
9. Overlay gorsellerini gosterme.
10. Model ve sonuclari zip olarak indirme.

Colab GPU tercih sirasi:

```text
A100 > L4 > T4
```

## 5. Egitim Plani

Ilk baseline komutu:

```bash
python scripts/train_yolo.py \
  --data configs/elements_dataset.yaml \
  --model yolo11s-seg.pt \
  --task segment \
  --name elements-seg-v1 \
  --epochs 100 \
  --imgsz 1024 \
  --device 0
```

Eger Colab T4 GPU bellegi yetersiz kalirsa alternatif:

```bash
python scripts/train_yolo.py \
  --data configs/elements_dataset.yaml \
  --model yolo11n-seg.pt \
  --task segment \
  --name elements-seg-v1-nano \
  --epochs 100 \
  --imgsz 768 \
  --device 0
```

## 6. Egitim Suresi Beklentisi

Gercek sure GPU turune, Colab yogunluguna, batch size'a ve image size'a gore degisir. Bu proje icin dataset kucuk oldugu icin ilk baseline makul surede tamamlanmalidir.

Tahmini sureler:

```text
A100, yolo11s-seg, imgsz 1024: yaklasik 10-25 dk
L4,   yolo11s-seg, imgsz 1024: yaklasik 20-45 dk
T4,   yolo11s-seg, imgsz 1024: yaklasik 35-90 dk
T4,   yolo11n-seg, imgsz 768:  yaklasik 15-45 dk
```

Bu sureler baseline icindir. Daha buyuk model, daha yuksek `imgsz`, daha fazla epoch veya daha fazla veri sureyi artirir.

## 7. Dogruluk ve Performans Degerleri

YOLO egitimi sonunda Ultralytics su metrikleri uretir:

```text
Precision
Recall
mAP50
mAP50-95
Box loss
Segmentation/mask loss
Classification loss
```

Segmentation icin en kritik metrikler:

```text
Mask mAP50
Mask mAP50-95
Precision
Recall
```

Anlamlari:

- `Precision`: Modelin yaptigi tespitlerin ne kadari dogru?
- `Recall`: Gercek etiketlerin ne kadarini yakaladi?
- `mAP50`: IoU 0.50 esiginde ortalama basari.
- `mAP50-95`: Daha zorlayici, 0.50 ile 0.95 IoU arasi ortalama basari.
- `Mask mAP`: Polygon/mask kalitesini olcer.

Kucuk dataset oldugu icin sadece tek sayiya bakmak dogru degildir. Metriklerle birlikte mutlaka gorsel overlay'ler de incelenmelidir.

## 8. Degerleri Nereden Kontrol Edecegiz?

Egitim bittikten sonra Colab veya lokal ortamda su klasor olusur:

```text
outputs/runs/elements-seg-v1/
```

Kontrol edilecek dosyalar:

```text
outputs/runs/elements-seg-v1/results.csv
outputs/runs/elements-seg-v1/results.png
outputs/runs/elements-seg-v1/confusion_matrix.png
outputs/runs/elements-seg-v1/val_batch*_pred.jpg
outputs/runs/elements-seg-v1/weights/best.pt
outputs/runs/elements-seg-v1/weights/last.pt
```

`results.csv`:

- Her epoch icin loss ve mAP degerlerini icerir.
- Final rapora gercek degerler buradan alinacaktir.

`results.png`:

- Loss ve metriklerin epoch boyunca nasil degistigini grafik olarak gosterir.

`confusion_matrix.png`:

- Hangi siniflarin birbirine karistigini gosterir.

`val_batch*_pred.jpg`:

- Validation goruntulerinde model tahminlerini gosterir.
- Mimari gozle kontrol icin en onemli ciktidir.

`best.pt`:

- En iyi validation skoruna sahip model agirligidir.
- Inference ve sonraki sistemlerde kullanilacak ana dosyadir.

## 9. Test Inference Kontrolu

Egitimden sonra test seti icin calistirilacak komut:

```bash
python scripts/infer_yolo.py \
  --weights outputs/runs/elements-seg-v1/weights/best.pt \
  --source data/roboflow/gtu-mimari-lejant/test/images \
  --task segment \
  --out outputs/reports/elements_test.json \
  --save-visuals \
  --device 0
```

Bu komut su ciktilari uretir:

```text
outputs/reports/elements_test.json
outputs/runs/infer-elements-test/
```

`elements_test.json` icinde her tespit su mantikta saklanir:

```json
{
  "label": "ahsap_dograma",
  "class_id": 0,
  "confidence": 0.91,
  "bbox_xyxy": [120.0, 80.0, 260.0, 210.0],
  "polygon": [[121.0, 82.0], [258.0, 84.0]]
}
```

## 10. Raporlanacak Gercek Egitim Sonuclari

Egitim tamamlandiktan sonra bu bolum doldurulacak.

```text
Calistirilan ortam:
GPU:
Model:
Epoch:
Image size:
Batch:
Toplam egitim suresi:

Final Precision:
Final Recall:
Final Box mAP50:
Final Box mAP50-95:
Final Mask mAP50:
Final Mask mAP50-95:
En iyi epoch:
```

Sinif bazli gozlemler:

```text
En iyi ayristirilan siniflar:
Karisan siniflar:
Eksik yakalanan siniflar:
Etiket kalitesi iyilestirme notlari:
```

## 11. Degistirdigimiz / Karar Verdiginiz Noktalar

Bu calismada alinan ana kararlar:

1. Export format olarak `YOLOv11` secildi.
2. Problem `Instance Segmentation` olarak ele alindi.
3. VLM fine-tuning ilk asamaya alinmadi.
4. Ilk baseline icin `YOLO segmentation` modeli secildi.
5. Model olarak baslangicta `yolo11s-seg.pt` belirlendi.
6. Dataset icin 42/5/5 train/valid/test split olusturuldu.
7. Buyuk veri ve model dosyalari GitHub'a eklenmedi.
8. Colab notebook ile tekrar edilebilir egitim akisi kuruldu.
9. Raporlama katmani JSON ciktisi uzerinden tasarlandi.

## 12. Sonraki Iyilestirme Adimlari

Ilk egitimden sonra su adimlar izlenecek:

1. `results.csv` ve `results.png` incelenecek.
2. `val_batch*_pred.jpg` ve test overlay'leri mimari gozle kontrol edilecek.
3. Sinif bazinda karisan elemanlar belirlenecek.
4. Gerekirse etiketler Roboflow'da duzeltilecek.
5. Veri artirilacaksa once az gorunen siniflara odaklanilacak.
6. Ikinci egitimde model/parametre degisikligi yapilacak.

Muhtemel ikinci deneyler:

```text
Deney A: yolo11n-seg, imgsz 768, hizli kontrol
Deney B: yolo11s-seg, imgsz 1024, ana baseline
Deney C: yolo11m-seg, imgsz 1024, daha yuksek kapasite
Deney D: yolo11s-seg, imgsz 1280, daha detayli maskeler
Deney E: yolo11s-seg, controlled augmentation
```

## 12.1 Kontrollu Augmentation Deneyi

Ultralytics varsayilan egitiminde augmentation zaten aciktir. Ancak mimari cephe segmentasyonu icin `mosaic=1.0` ve `erasing=0.4` gibi agresif ayarlar maske sinirlarini bozabilir. Bu nedenle ikinci deneyde daha kontrollu augmentation denenir:

```bash
yolo segment train \
  model=yolo11s-seg.pt \
  data=/content/lejanter_doga_vlm_codex/configs/elements_colab.yaml \
  project=/content/lejanter_doga_vlm_codex/outputs/runs \
  name=elements-seg-v2-aug-controlled \
  epochs=300 \
  patience=150 \
  imgsz=1024 \
  batch=-1 \
  device=0 \
  mosaic=0.3 \
  erasing=0.0 \
  scale=0.3 \
  translate=0.05 \
  hsv_h=0.01 \
  hsv_s=0.4 \
  hsv_v=0.3 \
  fliplr=0.5 \
  flipud=0.0 \
  perspective=0.0
```

Bu deney `elements-seg-v1` sonucunu ezmez; `elements-seg-v2-aug-controlled` adiyla ayri kaydedilir. Karsilastirma `Mask mAP50`, `Mask mAP50-95`, `Recall`, `results.png` ve validation/test overlay gorselleri uzerinden yapilacaktir.

## 13. Cephe Segmentasyonu ile Iliski

Mevcut dataset mimari malzeme/eleman siniflari icindir. Cephe bazli rapor icin iki secenek vardir:

1. Mevcut eleman segmentlerini dogrudan saymak.
2. Ayrica `facade_region` sinifi ile cephe yuzeylerini etiketlemek.

Cephe region etiketleri eklenirse pipeline su hale gelir:

```text
facade_region maskeleri
  -> eleman segmentleri
  -> spatial join
  -> "hangi eleman hangi cephede?" raporu
```

Bu, raporu daha mimari hale getirir:

```text
Sol cephede 14 ahsap_dograma, 6 cam, 3 demir eleman tespit edildi.
Sag cephede 8 tugla_duvar ve 2 dogal_tas_duvar_harcli alan tespit edildi.
```

## 14. Riskler ve Dikkat Edilecekler

- Dataset kucuk oldugu icin model overfit olabilir.
- Valid/test setleri sadece 5'er goruntu oldugu icin metrikler oynak olabilir.
- Bazi siniflar gorsel olarak birbirine yakin oldugundan confusion matrix dikkatle incelenmelidir.
- Dosya adlarinda Turkce karakterler Colab/Linux ortaminda sorun cikarmazsa devam edilebilir; sorun cikarsa Roboflow export adlari sadeleştirilebilir.
- Mimari rapor katmaninda sayilar model JSON'undan gelmeli, LLM'in sayi uydurmasina izin verilmemelidir.

## 15. Ozet

Bu asamada proje icin tekrar edilebilir bir segmentation egitim hatti kurulmustur. Roboflow verisi YOLOv11 instance segmentation formatinda alinmis, yerelde train/valid/test split'i hazirlanmis, egitim ve inference scriptleri yazilmis, Colab'da calisacak notebook eklenmistir.

Bir sonraki kritik adim Colab uzerinde ilk `elements-seg-v1` egitimini kosmak, `best.pt`, `results.csv`, `results.png` ve inference gorsellerini indirip gercek metriklerle bu raporu guncellemektir.
