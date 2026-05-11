# How to run this project (CS 445 / submission)

**Use this folder as working directory.** Run every command **after** `cd` into **`CS445_Final_Project`**. Paths in YAML such as `dataset/moving-gif-processed/...` are read **relative to that shell location**.

Put MGIF-style data here:

- `dataset/moving-gif-processed/moving-gif/` (`train/` and `test/`) for 256 configs  
- `dataset/moving-gif-processed/moving-gif-128/` for 128 configs  

If those folders are missing or misplaced, training will fail when the loader builds the index.


---

## Package layout

- **`cli.py`**, **`run.py`**, **`demo.py`** — launchers.  
- **`config/`** — YAML experiments.  
- **`scripts/`** — metrics helper (`eval_reconstruction_metrics.py`).  
- **`fom/modules/`** — generator, discriminator, keypoints / dense motion.  
- **`fom/data/`** — decoding (`video_io`), augmentations, datasets.  
- **`fom/services/`** — training loop and reconstruction / animation.  
- **`fom/entry.py`** — CLI routing.

---

## 1. Prerequisites

- **Python 3.8+** (`fom/entry.py` enforces 3.8+; tested with **3.9** and pinned `requirements.txt`)
- **GPU:** strongly recommended for longer training; CPU works but is slow per epoch.
- **FFmpeg:** optional for MP4; GIF configs avoid needing it (see `config/mgif-e2e-smoke.yaml`).

---

## 2. One-time setup

```bash
cd /path/to/CS445_Final_Project
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Training

**Local 256 MGIF (smaller batch):**

```bash
source .venv/bin/activate
python cli.py train --config config/mgif-256-local.yaml --log_dir logs
```

**Large-batch 256 (heavy VRAM, multi-device optional):**

```bash
python cli.py train --config config/mgif-256.yaml --log_dir logs --device_ids 0,1
```

**Smoke (128×128, 1 epoch, GIF recon/animate budgets):**

```bash
python cli.py train --config config/mgif-e2e-smoke.yaml --log_dir logs/e2e
```

Smoke is slow on CPU (often **~10–15+ minutes per epoch** for `mgif-e2e-smoke`; GPUs shorten this).

**Multi-epoch 128** (5 epochs, batch 16; aligns with the report’s main numbers):

```bash
python cli.py train --config config/mgif-128-5ep-run.yaml --log_dir logs/mgif128_5ep_run
```

**E2E smoke chain** (train → reconstruction → animate → metrics). After training, open the timestamped folder under `--log_dir`, note its exact name (spaces allowed), and **quote** `CKPT`:

```bash
source .venv/bin/activate
cd /path/to/CS445_Final_Project
python cli.py train --config config/mgif-e2e-smoke.yaml --log_dir logs/e2e_verify_run
CKPT="logs/e2e_verify_run/mgif-e2e-smoke <DATE_TIME>/00000000-checkpoint.pth.tar"
python cli.py reconstruction --config config/mgif-e2e-smoke.yaml --checkpoint "$CKPT"
python cli.py animate --config config/mgif-e2e-smoke.yaml --checkpoint "$CKPT"
python scripts/eval_reconstruction_metrics.py \
  --config config/mgif-e2e-smoke.yaml --checkpoint "$CKPT" \
  --max_videos 50 --out_csv logs/e2e_verify_run/metrics.csv
```

**Equivariance-off variant** (`mgif-ablation-no-equiv.yaml`) — optional comparison train:

```bash
python cli.py train --config config/mgif-ablation-no-equiv.yaml --log_dir logs/e2e
```

**Resume:**

```bash
python cli.py train --config config/mgif-256-local.yaml --log_dir logs \
  --checkpoint "logs/<run_folder>/00000010-checkpoint.pth.tar"
```

**Outputs:** under `--log_dir`, one timestamped folder per run with copied YAML, `log.txt`, checkpoints (`*-checkpoint.pth.tar`), `train-vis/`.

---

## 4. Reconstruction

`--config` must match how the checkpoint was trained.

```bash
python cli.py reconstruction \
  --config config/mgif-e2e-smoke.yaml \
  --checkpoint "logs/e2e_verify_run/mgif-e2e-smoke 10_05_26_18.01.14/00000000-checkpoint.pth.tar"
```

**Outputs:** `reconstruction/` and `reconstruction/png/` beside that checkpoint.

---

## 5. Animation

```bash
python cli.py animate \
  --config config/mgif-e2e-smoke.yaml \
  --checkpoint "logs/e2e_verify_run/mgif-e2e-smoke 10_05_26_18.01.14/00000000-checkpoint.pth.tar"
```

**Outputs:** `animation/` and `animation/png/` beside that checkpoint.

---

## 6. Reconstruction metrics (CSV)

```bash
python scripts/eval_reconstruction_metrics.py \
  --config config/mgif-e2e-smoke.yaml \
  --checkpoint "logs/e2e_verify_run/mgif-e2e-smoke 10_05_26_18.01.14/00000000-checkpoint.pth.tar" \
  --max_videos 50 --out_csv logs/e2e_verify_run/metrics_baseline.csv
```

---

## 7. Demo (still image + driving video)

```bash
python demo.py \
  --config config/mgif-256.yaml \
  --checkpoint /path/to/checkpoint.pth.tar \
  --source_image /path/to/source.png \
  --driving_video /path/to/driving.mp4 \
  --result_video /path/to/out.mp4 \
  --relative --adapt_scale
```

---

## 8. Other launch modes

```bash
python run.py --config config/mgif-256-local.yaml --mode train --log_dir logs
```

```bash
PYTHONPATH=. python -m fom --config config/mgif-256-local.yaml --mode train --log_dir logs
```

---

## 10. Troubleshooting

| Symptom | Likely fix |
|---------|------------|
| Dataset missing / empty index | Run commands **from** `CS445_Final_Project` with **`dataset/`** laid out as above; or edit YAML paths to match where MGIF tree lives |
| MP4 / ffmpeg errors | Install FFmpeg or switch configs to `.gif` |
| CUDA OOM | Reduce `batch_size` or use `mgif-256-local.yaml` / 128 YAML |
| `fom` import errors | Run **`python cli.py`** from this folder or export **`PYTHONPATH=.`** |
| Checkpoint mismatch | Same **`--config`** family as training |

---

## Citation

Method: Siarohin et al., *First Order Motion Model for Image Animation*, NeurIPS 2019.
