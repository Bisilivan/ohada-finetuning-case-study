# Protocole d'entraînement et de génération

Description complète, indépendante de l'article, pour quiconque veut reproduire ou auditer l'expérience.

## Modèle de base

`meta-llama/Llama-3.2-3B-Instruct`, chargé en bf16, sur un seul GPU (`device_map={"": 0}`).

## Configuration LoRA

```
r = 16
lora_alpha = 32
lora_dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
bias = "none"
task_type = "CAUSAL_LM"
```

Tous les modules de projection d'attention et du MLP sont ciblés. `modules_to_save` n'est volontairement pas utilisé : les embeddings de Llama 3.2 3B sont liés (`tie_word_embeddings=True`), les délier aurait ajouté environ 390M de paramètres pleinement entraînables et augmenté le risque de mémorisation pure sur seulement 114 exemples.

## Entraînement

```
num_train_epochs = 3
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
learning_rate = 2e-4
max_length = 3072
bf16 = true
gradient_checkpointing = true
```

`max_length` est fixé à 3072 après mesure directe : la séquence la plus longue du jeu d'entraînement (question + message système + réponse, tokenisée) atteint 2480 tokens, la moyenne est de 1215. Le jeu d'entraînement complet totalise donc environ 138 000 tokens (114 × 1215), vus trois fois au fil des trois epochs.

Matériel : un seul GPU T4 (palier gratuit de Kaggle). `CUDA_VISIBLE_DEVICES` est explicitement restreint à un seul GPU au tout début du notebook, avant tout import de `torch` : sur une machine à deux GPU, le `Trainer` de `transformers` active sinon automatiquement `torch.nn.DataParallel`, ce qui produit une erreur de cohérence de device pour un modèle chargé via `device_map`.

Résultat mesuré : perte d'entraînement de 1,900496 à 1,192392 sur 45 étapes (3 epochs), descente régulière, aucun signal d'instabilité.

## Génération (témoin et fine-tuné)

Message système, strictement identique dans les données d'entraînement et à la génération :

> Tu es un assistant juridique spécialisé en droit OHADA, en particulier le droit commercial général (AUDCG). Tu réponds avec rigueur, en citant systématiquement les articles et sources pertinents, sans jamais inventer de référence.

```
do_sample = false   # décodage glouton, déterministe
max_new_tokens = 1200
```

Le décodage glouton élimine toute variabilité liée à un choix de graine aléatoire : à modèle et question identiques, la réponse générée est strictement reproductible. Les 40 réponses témoin sont générées avant tout entraînement, sur le modèle intact. Les 40 réponses fine-tuné utilisent ensuite `trainer.model` (l'adaptateur LoRA actif), sans rechargement, avec la même fonction de génération et les mêmes 40 questions.

## Jeu de test

40 items, zéro recoupement littéral avec les 114 exemples d'entraînement, vérifié programmatiquement.

| Type | n | | Difficulté | n |
|---|---:|---|---|---:|
| Qualification de cas (CP) | 24 | | Basique | 7 |
| Question-réponse directe (QR) | 11 | | Intermédiaire | 17 |
| Argument type conclusions (DEF) | 5 | | Avancé | 12 |
| | | | Expert | 4 |
| **Total** | **40** | | **Total** | **40** |

## Fichiers de ce dossier

- `references.jsonl` — les 40 questions et leurs réponses de référence validées (id, entry_type, category, difficulty, question, reference)
- `reponses_temoin.jsonl` — réponses du modèle brut aux 40 questions (id, question, answer)
- `reponses_finetune.jsonl` — réponses du modèle fine-tuné aux mêmes 40 questions, même format

Les trois fichiers partagent le même `id` par question, ce qui permet de les recouper directement.
