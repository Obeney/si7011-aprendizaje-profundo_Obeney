# Taller 1 — Regresión Logística (MLP / DNN)

**Estudiante:** Obeney Londoño — 1017170826  
**Curso:** SI7011 Deep Learning  
**Dataset:** [Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## Objetivo

Implementar regresión logística en PyTorch para clasificar radiografías de tórax (NORMAL vs PNEUMONIA), como paso previo al taller de redes densas (MLP).

## Estructura

```
Taller1_MLP/
├── notebook/taller.ipynb    # Notebook ejecutado (entrega principal)
├── src/                     # Código modular reutilizable
├── outputs/metrics/         # Métricas exportadas
└── requirements.txt
```

## Resultados (test set)

| Métrica | Valor |
|---------|-------|
| Accuracy | 83.17% |
| F1 (weighted) | 0.83 |
| F1 (PNEUMONIA) | 0.87 |
| Baseline (siempre PNEUMONIA) | 62.5% |

**Errores:** 67 falsos positivos, 38 falsos negativos.

## Ejecución en Kaggle

1. Subir o vincular este repositorio como dataset en Kaggle.
2. Abrir `notebook/taller.ipynb`.
3. Activar **GPU** en Settings.
4. Verificar que el dataset `chest-xray-pneumonia` esté adjunto.

Ruta del dataset en Kaggle:

```
/kaggle/input/datasets/paultimothymooney/chest-xray-pneumonia/chest_xray
```

## Reflexión

1. La accuracy supera el baseline mayoritario gracias al aprendizaje supervisado, aunque el modelo sigue siendo lineal.
2. Predomina el error FP; en contexto clínico el FN es más crítico.
3. Con ~150k parámetros y poco dato, aparece sobreajuste (val loss > train loss).
4. El F1 confirma un balance razonable entre precision y recall en PNEUMONIA.

## Kaggle

- Notebook original del curso: [si7011-dl-mlp-pytorch](https://www.kaggle.com/code/juanmartinezv4399/si7011-dl-mlp-pytorch)
- Versión del estudiante: [si7011-dl-logreg-pytorch](https://www.kaggle.com/code/obeney18/si7011-dl-logreg-pytorch)
