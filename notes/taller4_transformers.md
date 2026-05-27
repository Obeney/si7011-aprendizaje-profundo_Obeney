# Taller 4 — TweetEval Transformers

- 4 clases: anger, joy, optimism, sadness.
- Mejor modelo: BERTweet (84.10% acc, F1 macro 0.8112).
- LoRA r=8: menos parámetros, rendimiento ligeramente inferior al fine-tune completo.
- Deploy: FastAPI + `Obeney/tweeteval-emotion-bertweet`.
