# Taller 1 — MLP (Intel Image Classification)

**Estudiante:** Obeney Londoño — 1017170826  
**Curso:** SI7011 Deep Learning  
**Dataset:** [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

## Objetivo

Entrenar un **MLP** (red densa) sobre imágenes aplanadas para clasificar escenas naturales en 6 clases y generar predicciones para el conjunto de competencia.

## Resultados

| Métrica | Valor |
|---------|-------|
| Train loss (época 10) | 0.6089 |
| Validation accuracy | 54.0% |
| Test accuracy | 53.0% |
| Predicciones competencia | 200 (`predictions.csv`) |

## Arquitectura

```
Input (150×150×1 → 22500) → Linear(512) → ReLU → Linear(128) → ReLU → Linear(6)
```

- **Loss:** CrossEntropyLoss  
- **Optimizer:** Adam, lr=1e-3  
- **Épocas:** 10 | **Batch size:** 32

## Estructura

```
Taller1_MLP/
├── notebook/taller.ipynb       # Notebook ejecutado (Kaggle)
├── src/                        # Código modular
├── outputs/metrics/
│   ├── results.json
│   └── predictions.csv
└── requirements.txt
```

## Ejecución en Kaggle

1. Adjuntar dataset `intel-image-classification`.
2. Activar **GPU T4 ×2**.
3. Ejecutar `notebook/taller.ipynb`.

Rutas en Kaggle:

```
/kaggle/input/datasets/puneet6060/intel-image-classification/seg_train/seg_train
/kaggle/input/datasets/puneet6060/intel-image-classification/seg_test/seg_test
/kaggle/input/datasets/puneet6060/intel-image-classification/seg_pred/seg_pred
```

## Análisis breve

- El modelo **reduce la loss** en entrenamiento pero la accuracy en val/test se estabiliza ~53–54% → generalización limitada y posible sobreajuste.
- Aplanar píxeles **ignora la estructura espacial** de la imagen, lo que limita la capacidad del MLP para capturar patrones locales.
- El muestreo **estratificado** garantiza representación balanceada en val/test (100 muestras c/u).

## Kaggle

- Notebook del estudiante: [notebook35f0564eb0](https://www.kaggle.com/code/obeney18/notebook35f0564eb0)
