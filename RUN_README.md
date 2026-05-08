# How to run this project (CS 445 / submission)

**Working directory matters.** Paths such as `dataset_params.root_dir: dataset/moving-gif-processed/...` in YAML are resolved relative to the **shell’s current working directory** (where you run `python …`), not relative to `cli.py`. Easiest options:

1. **`cd` into `CS445_Final_Project/`** and keep a **`dataset/`** folder here (copy, symlink, or unzip MGIF under `dataset/moving-gif-processed/…`), matching the default configs; or  
2. Stay in the **parent Git repo root** (`first-order-model/`) if your data already lives at **`./dataset/...`** there, and invoke the launcher with an explicit path, e.g. `python CS445_Final_Project/cli.py train --config CS445_Final_Project/config/mgif-e2e-smoke.yaml --log_dir CS445_Final_Project/logs/e2e`; or  
3. Edit **`root_dir`** in YAML to an **absolute path** or to something like **`../dataset/moving-gif-processed/moving-gif-128`** if your layout differs.

If your submission system expects a file named **`README.md`**, rename or copy this file to `README.md` (content can stay identical).

**Layout:** Launchers **`cli.py`**, **`run.py`**, **`demo.py`** sit at this folder’s root; **`config/`** holds YAML; **`scripts/`** holds metrics helpers. Package **`fom/`** is organized as:

- **`fom/modules/`** - generator, discriminator, keypoint / dense-motion blocks (**kept compatible** with the public FOMM checkpoints).  
- **`fom/data/`** - clip decoding (`video_io`), augmentations (`spatial_augmentation`), PyTorch datasets (`datasets`).  
- **`fom/services/`** - training loop, reconstruction, and animation (`training_loop`, `inference`, `model_factory`).  
- **`fom/entry.py`** - argparse, log dirs, dispatch.  
- **Legacy filenames** (`fom/train.py`, `fom/frames_dataset.py`, `fom/augmentation.py`, …) are thin re-exports so older imports still work.

Launchers prepend **this** directory to `sys.path` so `import fom` works.

**Course report:** when this tree lives inside the **`first-order-model`** repo, the Markdown report is at the repo root: **`../CS445_PROJECT_REPORT.md`**. Figures in that report point to **`report_assets/`** here.

---

## 1. Prerequisites

- **Python 3.8+** (`fom/entry.py` enforces 3.8+; tested with **3.9** and pinned `requirements.txt`)
- **Git** (optional)
- **GPU:** optional but strongly recommended for training. CPU runs are very slow for full epochs.
- **FFmpeg:** optional for **MP4** outputs. If `reconstruction` / `animate` fail with “no ffmpeg exe”, either install system FFmpeg or set configs to use **`.gif`** (see `config/mgif-e2e-smoke.yaml`).

---

## 2. One-time setup

```bash
cd /path/to/CS445_Final_Project
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Data:** place the processed MGIF-style tree so it matches `dataset_params.root_dir` **from your chosen cwd**, for example:

- `dataset/moving-gif-processed/moving-gif/` with `train/` and `test/` (256 configs)
- `dataset/moving-gif-processed/moving-gif-128/` for 128 configs

If those folders are missing **relative to cwd**, training will fail at dataset init with `FileNotFoundError` or an empty index.

---

## 3. Training

**Local 256 MGIF (smaller batch, fits single GPU better):**

```bash
source .venv/bin/activate
python cli.py train --config config/mgif-256-local.yaml --log_dir logs
```

**Paper-style 256 batch (needs large VRAM, often multi-GPU):**

```bash
python cli.py train --config config/mgif-256.yaml --log_dir logs --device_ids 0,1
```

**Fast pipeline smoke (128×128 MGIF, 1 epoch, small recon/animate budgets, GIF outputs):**

```bash
python cli.py train --config config/mgif-e2e-smoke.yaml --log_dir logs/e2e
```

Smoke training is **CPU-heavy** (about **10–15+ minutes per epoch** on a typical laptop CPU for `mgif-e2e-smoke`; much faster on GPU). Use GPU when available.

**Full E2E smoke in one session** (train, then recon, animate, metrics). After training, locate the new folder under `--log_dir` named like `mgif-e2e-smoke DD_MM_YY_HH.MM.SS/` and set `CKPT` to its `00000000-checkpoint.pth.tar` (**quote** the variable-names contain spaces):

```bash
source .venv/bin/activate
cd /path/to/CS445_Final_Project   # or stay at monorepo root and prefix paths as in note below
python cli.py train --config config/mgif-e2e-smoke.yaml --log_dir logs/e2e_verify_run
CKPT="logs/e2e_verify_run/mgif-e2e-smoke <DATE_TIME>/00000000-checkpoint.pth.tar"
python cli.py reconstruction --config config/mgif-e2e-smoke.yaml --checkpoint "$CKPT"
python cli.py animate --config config/mgif-e2e-smoke.yaml --checkpoint "$CKPT"
python scripts/eval_reconstruction_metrics.py \
  --config config/mgif-e2e-smoke.yaml --checkpoint "$CKPT" \
  --max_videos 50 --out_csv logs/e2e_verify_run/metrics.csv
```

**From monorepo root** (when `dataset/` is at **`./dataset/...`** next to **`CS445_Final_Project/`**):

```bash
cd /path/to/first-order-model
CFG=CS445_Final_Project/config/mgif-e2e-smoke.yaml
LOG=CS445_Final_Project/logs/e2e_verify_run
python3 CS445_Final_Project/cli.py train --config "$CFG" --log_dir "$LOG"
CKPT="$LOG/mgif-e2e-smoke <DATE_TIME>/00000000-checkpoint.pth.tar"
python3 CS445_Final_Project/cli.py reconstruction --config "$CFG" --checkpoint "$CKPT"
python3 CS445_Final_Project/cli.py animate --config "$CFG" --checkpoint "$CKPT"
python3 CS445_Final_Project/scripts/eval_reconstruction_metrics.py --config "$CFG" --checkpoint "$CKPT" \
  --max_videos 50 --out_csv "$LOG/metrics.csv"
```

**Equivariance ablation (same smoke structure, equivariance losses off):**

```bash
python cli.py train --config config/mgif-ablation-no-equiv.yaml --log_dir logs/e2e
```

**Resume** from a checkpoint (same config as that run):

```bash
python cli.py train --config config/mgif-256-local.yaml --log_dir logs \
  --checkpoint "logs/<run_folder>/00000010-checkpoint.pth.tar"
```

**Outputs:** a timestamped folder under `--log_dir`, for example `logs/mgif-256-local 10_05_26_12.00.00/`, containing:

- copied YAML, `log.txt`, `*-checkpoint.pth.tar`, `train-vis/` (PNG grids)

---

## 4. Reconstruction (needs a checkpoint)

`--config` must match the **architecture** used to train that checkpoint (resolution, `num_kp`, etc.).

```bash
python cli.py reconstruction \
  --config config/mgif-e2e-smoke.yaml \
  --checkpoint "logs/e2e_verify_run/mgif-e2e-smoke 10_05_26_18.01.14/00000000-checkpoint.pth.tar"
```

(Replace with any run folder that contains your checkpoint; **quote** paths with spaces.)

**Outputs:** next to the checkpoint file: `reconstruction/` (GIF or MP4 per config) and `reconstruction/png/` (wide strip PNGs).

---

## 5. Animation (needs a checkpoint)

```bash
python cli.py animate \
  --config config/mgif-e2e-smoke.yaml \
  --checkpoint "logs/e2e_verify_run/mgif-e2e-smoke 10_05_26_18.01.14/00000000-checkpoint.pth.tar"
```

**Outputs:** `animation/` and `animation/png/` beside that checkpoint.

Smoke configs set `max_driving_frames` so CPU runs finish in reasonable time.

---

## 6. Reconstruction metrics (CSV, no GIF IO)

```bash
python scripts/eval_reconstruction_metrics.py \
  --config config/mgif-e2e-smoke.yaml \
  --checkpoint "logs/e2e_verify_run/mgif-e2e-smoke 10_05_26_18.01.14/00000000-checkpoint.pth.tar" \
  --max_videos 50 --out_csv logs/e2e_verify_run/metrics_baseline.csv
```

---

## 7. Demo (single source image + driving video)

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

## 8. Alternative entry (`run.py`, module)

```bash
python run.py --config config/mgif-256-local.yaml --mode train --log_dir logs
```

```bash
PYTHONPATH=. python -m fom --config config/mgif-256-local.yaml --mode train --log_dir logs
```

---

## 9. Layout (what to zip for submission)

When this folder is part of the full **first-order-model** Git repository, the final report markdown lives one level up: **`../CS445_PROJECT_REPORT.md`** (repo root). Figures referenced there are under **`CS445_Final_Project/report_assets/`** from the repo’s perspective (this folder’s **`report_assets/`**).

| Path | Purpose |
|------|--------|
| `fom/` | Package root: **`modules/`** (nets), **`data/`** (I/O + datasets), **`services/`** (train + inference), **`entry.py`**, shims |
| `cli.py`, `run.py`, `demo.py` | Launchers |
| `config/*.yaml` | Hyperparameters and data paths |
| `scripts/` | `eval_reconstruction_metrics.py`, shell helpers |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Optional; keeps venv and caches out of version control |
| `RUN_README.md` | This file (primary run instructions for graders) |
| `report_assets/` | Default figures for the report (PNG + GIF); replace with your own `logs/...` exports when available |
| `dataset/` | Processed MGIF tree (`moving-gif-processed/...`); large-often omitted from zip with a Drive link |

**Large artifacts:** you may omit **`dataset/`** (processed MGIF files) and **`logs/`** from the zip if the report links to data on Drive; graders then need your data link to re-run training.

---

## 10. Troubleshooting

| Symptom | Likely fix |
|---------|------------|
| `FileNotFoundError` / empty dataset | **`dataset/`** must exist **relative to cwd** (see top of this file); fix cwd, symlink, or set `root_dir` to an absolute path |
| MP4 / ffmpeg error | Install FFmpeg, or set `reconstruction_params.format` / `animate_params.format` to `.gif` |
| CUDA OOM | Lower `batch_size` in YAML or use `mgif-256-local.yaml` / 128 config |
| Import / `fom` not found | Use `python cli.py` (it prepends this folder) or set `PYTHONPATH=.` |
| Checkpoint load mismatch | `--config` must match how the checkpoint was trained |

---

## Citation

Method: Siarohin et al., *First Order Motion Model for Image Animation*, NeurIPS 2019. Official code: `https://github.com/AliaksandrSiarohin/first-order-model`.
