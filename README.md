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
│   ├── ECGenv.yml              # Conda environment
│   ├── data/                   # H5 data files (auto-created by preprocess.py)
│   ├── csv/                    # Evaluation metric CSVs (auto-created)
│   ├── gradcam/                # Grad-CAM output images (auto-created)
│   ├── models/                 # Model checkpoints (auto-created)
│   ├── record_history/         # Training metrics (auto-created)
│   └── ecg-preprocessing-main/ # Signal preprocessing tool
├── ecg-image-generator/        # ECG image synthesis toolkit
├── ptb-xl-dataset/             # PTB-XL dataset (download separately, see Step 1)
├── output-images/              # Generated ECG PNG images (see Step 2)
└── Project_in_ECG.pdf          # Project report
```

---

## Setup

Run the one-command setup (Windows, requires Anaconda):
```bat
setup.bat
```

This installs `nb_conda_kernels` in the base environment and creates the `ECGenv` conda environment. After setup, launch JupyterLab from the base environment:
```bat
jupyter lab
```
Select the **ECGenv** kernel inside JupyterLab.

---

## Pipeline

Run the following steps in order. **Steps 1–3 are run from the root directory in Anaconda Prompt.**

### Step 1 — Download PTB-XL

Download the dataset (~3 GB uncompressed) using one of the following methods, then place the files at `ECG-classifier/ptb-xl-dataset/`:

**Option A — wget** (Linux/macOS/WSL):
```bash
wget -r -N -c -np --cut-dirs=4 --directory-prefix=ptb-xl-dataset https://physionet.org/files/ptb-xl/1.0.3/
```

**Option B — AWS CLI**:
```bash
aws s3 sync --no-sign-request s3://physionet-open/ptb-xl/1.0.3/ ptb-xl-dataset/
```

**Option C — ZIP**: Download the ZIP file (1.7 GB) from https://physionet.org/content/ptb-xl/1.0.3/, extract it, and rename/move the extracted folder to `ptb-xl-dataset/` inside the project root.

The folder should contain:
```
ptb-xl-dataset/
    ptbxl_database.csv
    scp_statements.csv
    RECORDS_LowRes.txt
    records100/
    records500/
```

### Step 2 — Generate ECG images

```bash
conda activate ECGenv
cd ecg-image-generator
python gen_ecg_images_from_data_batch.py -i ../ptb-xl-dataset/records100 -o ../output-images/ -r 30
```

Images are named `{ecg_id}_*.png`.
The flag `-r 30` determines the resolution of images in DPI.

### Step 3 — Generate all H5 data files

```bash
cd Notepad
python preprocess.py --dataset S   # small subset (2000 signal / 500 image)
python preprocess.py --dataset L   # full dataset (~22k records)
```

This generates:
- `signal_train_*.h5` / `signal_test_*.h5` — resampled ECG signals + labels
- `train_*.h5` / `test_*.h5` — grayscale image arrays + labels

Files that already exist are skipped automatically.

### Step 4 — Run the Signal notebook

Open `Notepad/Signal based notepad.ipynb` and run all cells top to bottom.

This will:
1. Load H5 files generated in Step 3 (skipped if already done)
2. Train the ResNet-1D signal model → saves `models/signal_model_{iter}.pth`
3. Evaluate on the test set
4. **Append `signal_logits` to `train_*.h5`** (required for Step 5 KD training)

Hyperparameters can be changed in cell 3 (defaults from report used).

### Step 5 — Run the Image notebook

Open `Notepad/Image based notepad.ipynb` and run all cells top to bottom.

This will:
1. Load H5 files (skipped if already created by Step 3)
2. Train ResNet-50 **with KD** (`iter_kd`), then immediately **without KD** (`iter_nokd`), back to back
3. Save metrics for both runs to `record_history/metrics.h5`
4. Evaluate and compare against the signal baseline

Hyperparameters can be changed in cell 3 (defaults from report used).

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

Re-running with the same iter will retrain the model and overwrite both the saved metrics and the `.pth` checkpoint for that iter.

After a new run, update the iter values in `metrics.ipynb` Cell 2 to point at the new results.

---

## Key Design Choices

- **Knowledge distillation loss:** `alpha * BCE(logits, labels) + beta * T^2 * BCE(sigmoid(logits/T), sigmoid(signal_logits/T))`  with `alpha=1, beta=1, T=3`
- **Image preprocessing:** grayscale, top 27.5% cropped (header), normalised to [0, 1]
- **Signal preprocessing:** resampled to 400 Hz, zero-padded to 4096 samples, baseline removed
- **Train/val/test split:** PTB-XL stratified fold 10 = test, 80/20 random split of folds 1-9 = train/val
