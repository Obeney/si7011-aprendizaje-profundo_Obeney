# Taller 2 — CNN Tiny ImageNet (Transfer Learning)

**Estudiante:** Obeney Londoño — 1017170826  
**Curso:** SI7011 Deep Learning  
**Dataset:** [Tiny ImageNet](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet)

## Objetivo

Clasificar **200 clases** de Tiny ImageNet usando **ResNet34** preentrenado en ImageNet, con augmentación, fine-tuning y generación de submission para el conjunto test.

## Resultados

| Métrica | Valor |
|---------|-------|
| Mejor val accuracy | **72.08%** |
| Train accuracy (ép. 10) | 81.19% |
| Épocas | 10 |
| Modelo | ResNet34 + Dropout + Linear(200) |

## Arquitectura

- Base: `resnet34` (weights ImageNet1K_V1)
- Cabeza: `Dropout(0.3)` → `Linear(512, 200)`
- Pérdida: `CrossEntropyLoss(label_smoothing=0.1)`
- Optimizador: `AdamW` (lr=1e-4, weight_decay=1e-4)

## Estructura

```
Taller2_TinyImageNet/
├── notebook/taller.ipynb
├── src/
├── outputs/metrics/results.json
└── requirements.txt
```

## Datos en Kaggle

CSV desde Google Drive (train/val/test) + imágenes:

```
/kaggle/input/datasets/akash2sharma/tiny-imagenet/tiny-imagenet-200/train/
/kaggle/input/datasets/akash2sharma/tiny-imagenet/tiny-imagenet-200/val/images/
```

## Ejecución

1. Adjuntar dataset Tiny ImageNet en Kaggle.
2. Activar **GPU T4 ×2**.
3. Ejecutar `notebook/taller.ipynb`.
4. Generar `submission.csv` para el test set (sin labels).

## Análisis breve

- Transfer learning acelera convergencia: val accuracy pasa de ~65% (ép. 1) a ~72% (ép. 10).
- Brecha train–val (~9 pp) sugiere **sobreajuste leve** pese a augmentación y regularización.
- Augmentación fuerte es clave al escalar imágenes 64×64 → 224×224.

## Kaggle

- [Taller2 — obeney18](https://www.kaggle.com/code/obeney18/taller2)
