# Taller 4 — Transformers / NLP (TweetEval)

**Estudiante:** Obeney Londoño — 1017170826  
**Curso:** SI7011 Deep Learning  
**Dataset:** [TweetEval — emotion](https://huggingface.co/datasets/tweet_eval)

## Objetivo

Pipeline completo de clasificación de emociones en tweets: EDA → pipeline reutilizable → fine-tuning → LoRA → deployment API.

## Notebooks (orden de ejecución)

| # | Notebook | Contenido |
|---|----------|-----------|
| 1 | `part1_data.ipynb` | Carga, EDA, comparación de tokenizadores |
| 2 | `part2_pipeline.ipynb` | Funciones compartidas (`make_trainer`, etc.) |
| 3 | `part3_distilbert.ipynb` | Experimento A: DistilBERT |
| 4 | `part4_bertweet.ipynb` | Experimento B: BERTweet |
| 5 | `part5_lora.ipynb` | Bonus: LoRA sobre BERTweet |
| 6 | `part6_deployment.ipynb` | FastAPI + inferencia |

> Partes 3–5 ejecutan `%run part2_pipeline.ipynb` antes de entrenar.

## Resultados (test set)

| Modelo | Accuracy | Macro F1 |
|--------|----------|----------|
| DistilBERT | 80.08% | 0.7552 |
| **BERTweet** | **84.10%** | **0.8112** |
| BERTweet-LoRA | 82.76% | 0.7919 |

## Deployment

- Modelo en Hugging Face: `Obeney/tweeteval-emotion-bertweet`
- API REST con FastAPI (Parte 6)

## Estructura

```
Taller4_AdvancedCNN/
├── notebook/          # 6 notebooks ejecutados
├── src/               # pipeline modular
├── outputs/metrics/
└── requirements.txt
```

## Ejecución

**Lightning AI / Kaggle / local con GPU recomendada.**

```bash
pip install -r Taller4_AdvancedCNN/requirements.txt
# Ejecutar part1 → part2 → part3 → part4 → part5 → part6
```

## Referencia curso

- [SI7011 — Lecture06 exercise](https://github.com/jdmartinev/SI7011-DeepLearning/tree/main/Lecture06/notebooks/excercise)
