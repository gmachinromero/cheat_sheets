# Detoxify: Toxic Comment Classification with ⚡ Pytorch Lightning and 🤗 Transformers
___

## 1. Install library

```
pip install detoxify
```

```Python
from detoxify import Deoxify
```

## 2. Multilingual prediction

Pass a list (not other format allowed) of texts, and get a dictionary with the toxicity score of each text. Detoxify has three pre-trained models, which are listed with their labels scoring below:

- original: toxic, severe_toxic, obscene, threat, insult, identity_hate.
- unbiased: toxicity, severe_toxicity, obscene, threat, insult, identity_attack, sexual_explicit.
- multilingual: toxicity

*multilingual* model is the unique that can predict toxicity in a different language from english.

```Python
# Hate speech level prediction
# ------------------------------------------------------------------------------
results = Detoxify('multilingual').predict(list_name)
```

## 3. Bibliography

- https://github.com/unitaryai/detoxify