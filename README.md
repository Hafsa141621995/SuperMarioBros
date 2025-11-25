# SuperMarioBros
# Mario PPO – Projet IA / Renforcement (Ynov)

Ce repo contient mon projet d’apprentissage par renforcement profond sur **Super Mario Bros (World 1-1)**.  
L’objectif : entraîner un agent PPO capable de jouer au niveau 1-1 à partir des **images du jeu**, sans règles codées à la main.

Pour l’instant, l’agent :
- avance vers la droite,
- arrive à tuer quelques goombas,
- progresse un peu dans le niveau,
- mais **ne termine pas encore le stage** (il meurt avant le drapeau).

Ce comportement est cohérent avec le temps d’entraînement limité et le budget machine (CPU sous WSL, pas de GPU).

---

##  Tech & structure

- Environnement : `gym-super-mario-bros` + `nes-py`
- Niveau : `SuperMarioBros-1-1-v3` avec `apply_api_compatibility=True`
- Actions : `SIMPLE_MOVEMENT` (7 actions discrètes)
- Wrappers maison :
  - `compat.py` → gère la différence Gym 0.26 (reset/step)  
  - `src/env.py` → pré-traitement :
    - gris, resize 84×84, normalisation /255
    - `CustomSkipFrame` → empilement de 4 frames (state shape = `(4, 84, 84)`)
    - `CustomReward` → reward shaping basé sur le score, la progression et les morts
- Modèle : `src/model.py`
  - CNN 4×conv → dense 512
  - tête actor (policy) + tête critic (value), initialisation orthogonale
- Entraînement : `train.py`
  - PPO avec plusieurs environnements en parallèle (`MultipleEnvironments`)
  - logging TensorBoard dans `tensorboard/ppo_super_mario_bros/...`
- Évaluation :
  - `eval_gif.py` → recharge un checkpoint `.pt`, joue un épisode et génère un GIF / MP4
  - `eval_clean.py` → version plus “propre” en deux envs (step + render) pour des vidéos stables

---

##  Runs principaux (ce que j’ai vraiment lancé)

J’ai gardé deux runs “sérieux” pour le rapport :

### Run A – Baseline rapide

- `world = 1`, `stage = 1`, `action_type = "simple"`
- Hyperparamètres principaux :
  - `lr = 1e-4`
  - `gamma = 0.9`
  - `tau = 1.0`
  - `beta = 0.01` (entropie = exploration)
  - `epsilon = 0.2` (PPO clip)
- Budget :
  - `num_local_steps = 256`
  - `num_processes = 2`
  - `batch_size = 8`
  - `num_epochs = 3`
- Résultat (eval avec `eval_gif.py`) :
  - score total ≈ **2013**
  - Mario avance, tue quelques goombas, mais ne finit pas le niveau.

### Run B – Baseline “complète”

- mêmes hyperparamètres principaux (`lr`, `gamma`, `beta`, `epsilon`)
- budget plus lourd :
  - `num_local_steps = 512`
  - `num_processes = 8`
  - `batch_size = 16`
  - `num_epochs = 10`
- Résultat (eval `ppo_latest.pt`) :
  - score total ≈ **2022**
  - comportement très proche de Run A, légère meilleure progression mais toujours pas de clear.

---

##  Logs et visualisation

- TensorBoard :
  - dossier : `tensorboard/ppo_super_mario_bros/...`
  - tags utilisés :
    - `train/mean_reward_per_rollout`
    - `train/total_loss`
    - `loss/actor` / `loss/critic` (selon la version)
- Vidéos :
  - GIF/MP4 générés par `eval_gif.py` / `eval_clean.py`
  - rendu en 240×256 (NES), donc très “pixelisé” mais suffisant pour analyser le comportement

---

## ⚠️ Limitations connues

- Entraînement uniquement sur CPU (WSL), temps limité → pas de grid search massif d’hyperparamètres.
- Certains scripts d’évaluation affichent bien “mp4 saved” mais les fichiers ne sont pas toujours visibles dans le dossier attendu (probable souci de chemin WSL / Windows ou de ffmpeg).
- Problèmes de santé → plusieurs séances de cours manquées, rattrapage en autonomie à partir des PDFs et d’un repo GitHub existant.
- L’agent ne termine pas encore le niveau 1-1, mais montre un comportement **partiellement raisonnable** (progression + gestion de quelques ennemis).

---

## ▶️ Comment lancer un entraînement

Dans un environnement virtuel Python 3.9 avec les dépendances installées (`gym-super-mario-bros`, `torch`, etc.) :

```bash
python train.py \
  --world 1 --stage 1 --action_type simple
