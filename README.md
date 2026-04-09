# ECG Disease Classification: Signal vs. Image-Based Models

Comparison of two deep learning approaches for multi-label ECG diagnosis on the [PTB-XL dataset](https://physionet.org/content/ptb-xl/1.0.3/), with knowledge distillation from the signal model to the image model.

**5 diagnostic classes:** NORM, MI (Myocardial Infarction), STTC (ST/T Change), CD (Conduction Disturbance), HYP (Hypertrophy)

| Model | Macro AUROC | Macro Brier |
|---|---|---|
| Signal (ResNet-1D) | 0.928 | 0.085 |
| Image + KD (ResNet-50) | 0.917 | 0.091 |
| Image only (ResNet-50) | 0.900 | 0.099 |

See `Project_in_ECG.pdf` for the full report.

---

## Repository Structure

```
ECG-classifier/
├── Notepad/                    # Jupyter notebooks (run from here)
│   ├── Signal based notepad.ipynb
│   ├── Image based notepad.ipynb
│   ├── metrics.ipynb           # Training curves + report comparison plot
│   ├── preprocess.py           # Generates all H5 data files (run before notebooks)
│   ├── NoteEnv.yml             # Conda environment
│   ├── record_history/         # Training metrics (auto-created)
│   └── ecg-preprocessing-main/ # Signal preprocessing tool
├── ecg-image-generator/        # ECG image synthesis toolkit
├── ptb-xl-a-large-.../         # PTB-XL dataset (download separately)
├── output-images/              # Generated ECG PNG images (see Step 2)
└── Project_in_ECG.pdf          # Project report
```

Trained models are saved outside the project:
```
signal_MODEL/model_{iter}.pth   # Signal model checkpoints
IMAGE_MODEL/model_{iter}.pth    # Image model checkpoints (with KD)
IMAGE_MODEL/model_NoKD_{iter}.pth
```

---

## Setup

Run the one-command setup (Windows, requires Anaconda):
```bat
setup.bat
```

This installs `nb_conda_kernels` in the base environment and creates the `NoteEnv` conda environment. After setup, launch JupyterLab from the base environment:
```bat
jupyter lab
```
Select the **NoteEnv** kernel inside JupyterLab.

---

## Pipeline

Run the following steps in order. **Steps 3–5 are run from the `Notepad/` directory.**

### Step 1 — Download PTB-XL

Download from https://physionet.org/content/ptb-xl/1.0.3/ and place at:
```
ECG-classifier/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
    ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
        ptbxl_database.csv
        scp_statements.csv
        RECORDS_LowRes.txt
        records100/
        records500/
```

### Step 2 — Generate ECG images

```bash
cd ECG-classifier/ecg-image-generator
python gen_ecg_images_from_data_batch.py \
    --input_file ../ptb-xl-a-large-.../ptb-xl-a-large-.../ \
    --output_dir ../output-images/
```

Images are named `{ecg_id}_*.png`.

### Step 3 — Generate all H5 data files

Run once before the notebooks. Activate the conda environment first, then from `Notepad/`:
```bash
conda activate NoteEnv
python preprocess.py --dataset S   # small subset (2000 signal / 500 image)
python preprocess.py --dataset L   # full dataset (~19k records)
```

This generates:
- `signal_train_*.h5` / `signal_test_*.h5` — resampled ECG signals + labels
- `train_*.h5` / `test_*.h5` — grayscale image arrays + labels

Files that already exist are skipped automatically.

### Step 4 — Run the Signal notebook

Open `Notepad/Signal based notepad.ipynb` and run all cells top to bottom.

This will:
1. Load H5 files generated in Step 3 (skipped if already done)
2. Train the ResNet-1D signal model → saves `signal_model.pth` and `../../signal_MODEL/model_{iter}.pth`
3. Evaluate on the test set
4. **Append `signal_logits` to `train_*.h5`** (required for Step 5 KD training)

### Step 5 — Run the Image notebook

Open `Notepad/Image based notepad.ipynb` and run all cells top to bottom.

This will:
1. Load H5 files (skipped if already created by Step 3)
2. Train ResNet-50 **with KD** (`iter_kd`), then immediately **without KD** (`iter_nokd`), back to back
3. Save metrics for both runs to `record_history/metrics.h5`
4. Evaluate and compare against the signal baseline

### Step 6 — View metrics

Open `Notepad/metrics.ipynb` and run all cells.

- **Cell 1** — Training curve viewer: set `model = "signal"` or `"image"` and `iter` to inspect any run
- **Cell 2** — AUROC comparison plot across all three models, reading from the locally generated metrics files

---

## Re-running the pipeline

Steps 1–3 are fully idempotent — files that already exist are skipped automatically.

Training (Steps 4–5) always re-runs. To record results from a new run without overwriting previous ones, **change the iter values** in the parameter cell before running:

- Signal notebook Cell 2 (parameters): `iter`
- Image notebook Cell 3 (parameters): `iter_kd`, `iter_nokd`

Metrics are saved per iter key in the H5 files. Re-running with the same iter will retrain the model but not overwrite the saved metrics — delete the H5 file (or specific keys) first if you want to replace them.

After a new run, update the iter values in `metrics.ipynb` Cell 2 to point at the new results.

---

## Key Design Choices

- **Knowledge distillation loss:** `alpha * BCE(logits, labels) + beta * T^2 * BCE(sigmoid(logits/T), sigmoid(signal_logits/T))`  with `alpha=1, beta=1, T=3`
- **Image preprocessing:** grayscale, top 27.5% cropped (header), normalised to [0, 1]
- **Signal preprocessing:** resampled to 400 Hz, zero-padded to 4096 samples, baseline removed
- **Train/val/test split:** PTB-XL stratified fold 10 = test, 80/20 random split of folds 1-9 = train/val
