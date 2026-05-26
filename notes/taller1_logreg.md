# Notas del curso SI7011

Apuntes y decisiones técnicas por taller.

## Taller 1 — LogReg Chest X-Ray

- Dataset desbalanceado → comparar siempre contra baseline mayoritario (62.5%).
- Imagen 224×224×3 aplanada → 150.528 features; alto riesgo de overfitting.
- Augmentación solo en train; val/test sin transformaciones aleatorias.
- BCEWithLogitsLoss + Adam lr=1e-3, 5 épocas.
