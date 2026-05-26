# Taller 3 — RNN/LSTM (Series de tiempo)

**Estudiante:** Obeney Londoño — 1017170826  
**Curso:** SI7011 Deep Learning  
**Dataset:** Bike Sharing Demand (OpenML #44063)

## Objetivo

Predecir demanda horaria de bicicletas compartidas con un pipeline completo: preprocesamiento (Parte 1) + modelo recurrente GRU (Parte 2).

## Notebooks

| Parte | Archivo | Contenido |
|-------|---------|-----------|
| 1 | `notebook/part1_data.ipynb` | EDA, features, baselines, guardado de arrays |
| 2 | `notebook/part2_model.ipynb` | Dataset secuencial, GRU, entrenamiento, evaluación |

> **Orden de ejecución:** Parte 1 → Parte 2 (Parte 2 requiere `data/bike_processed/`).

## Resultados (MAE en test, bikes/hour)

| Modelo | Test MAE |
|--------|----------|
| Naive (t+1 = t) | 80.78 |
| HistGradientBoosting | 33.94 |
| **GRU (RNN)** | **34.45** |

## Arquitectura

```
Input [batch, 24, 28] → GRU(64) × 2 → Dropout → Linear(1)
```

- Ventana: 24 horas | Horizonte: 1 hora
- Loss: MSE sobre target normalizado
- Early stopping (patience=20) + ReduceLROnPlateau

## Estructura

```
Taller3_CNN/
├── notebook/
│   ├── part1_data.ipynb
│   └── part2_model.ipynb
├── src/
├── outputs/metrics/results.json
└── requirements.txt
```

## Ejecución

**Lightning AI / local con GPU recomendada para Parte 2.**

```bash
pip install -r Taller3_CNN/requirements.txt
jupyter notebook Taller3_CNN/notebook/part1_data.ipynb
jupyter notebook Taller3_CNN/notebook/part2_model.ipynb
```

## Análisis breve

- El GRU reduce el error más de un 50% respecto al baseline naive.
- Resultado casi equivalente al HistGradientBoosting (~0.5 bikes/hour de diferencia).
- La brecha train–val sugiere regularización adecuada con dropout y early stopping.

## Referencia curso

- [SI7011 — Lecture05 exercise](https://github.com/jdmartinev/SI7011-DeepLearning/tree/main/Lecture05/notebooks/excercise)
