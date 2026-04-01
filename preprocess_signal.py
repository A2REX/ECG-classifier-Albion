"""
preprocess_signal.py — build signal_train.h5 and signal_test.h5 from PTB-XL.

Replicates what ecg-preprocessing-main/generate_h5.py did in the original notebook:
  - reads records500 (500 Hz, 5000 samples, 12 leads)
  - resamples to 400 Hz  →  4000 samples
  - removes baseline wandering with a high-pass Butterworth filter at 0.5 Hz
  - pads to 4096 samples
  - saves as HDF5 with keys 'tracings' (N, 4096, 12) and 'labels' (N, 5)

Usage:
    python preprocess_signal.py data/ptbxl \
        --output_dir data/signal_hdf5

Falls back to records100 (100 Hz) if records500 is not present.
"""

import argparse
import ast
import os

import h5py
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, resample_poly
from tqdm import tqdm

SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
TARGET_FREQ   = 400
TARGET_LEN    = 4096


def highpass_baseline(signal, fs, cutoff=0.5, order=3):
    """Remove baseline wander with a high-pass Butterworth filter."""
    nyq = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype='highpass')
    return filtfilt(b, a, signal, axis=0)


def process_record(path, src_freq):
    """Load one WFDB record, resample to 400 Hz, remove baseline, pad to 4096."""
    record = wfdb.rdrecord(path)
    sig = record.p_signal.astype(np.float32)          # (n_samples, n_leads)

    # Resample to TARGET_FREQ
    if src_freq != TARGET_FREQ:
        from math import gcd
        g = gcd(TARGET_FREQ, src_freq)
        sig = resample_poly(sig, TARGET_FREQ // g, src_freq // g, axis=0).astype(np.float32)

    # Baseline removal
    sig = highpass_baseline(sig, TARGET_FREQ).astype(np.float32)

    # Pad or truncate to TARGET_LEN
    n = len(sig)
    if n < TARGET_LEN:
        sig = np.pad(sig, ((0, TARGET_LEN - n), (0, 0)))
    else:
        sig = sig[:TARGET_LEN]

    return sig  # (TARGET_LEN, n_leads)


def build_labels(df, agg_df):
    """Multi-hot label matrix (N, 5) from PTB-XL scp_codes column."""
    def row_label(scp_codes):
        lbl = np.zeros(len(SUPERCLASSES), dtype=np.float32)
        for code in scp_codes.keys():
            if code in agg_df.index:
                sc = agg_df.loc[code, 'diagnostic_class']
                if sc in SUPERCLASSES:
                    lbl[SUPERCLASSES.index(sc)] = 1.0
        return lbl

    return np.stack([row_label(r) for r in df['scp_codes']])


def write_hdf5(path, tracings, labels):
    with h5py.File(path, 'w') as f:
        f.create_dataset('tracings', data=tracings, compression='gzip')
        f.create_dataset('labels',   data=labels,   compression='gzip')
    print(f'Saved {path}  shape={tracings.shape}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir',  help='PTB-XL root (contains ptbxl_database.csv)')
    parser.add_argument('--output_dir', default='data/signal_hdf5',
                        help='directory for output HDF5 files (default: data/signal_hdf5)')
    parser.add_argument('--test_fold', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load metadata ---
    df = pd.read_csv(os.path.join(args.data_dir, 'ptbxl_database.csv'), index_col='ecg_id')
    df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)

    scp = pd.read_csv(os.path.join(args.data_dir, 'scp_statements.csv'), index_col=0)
    agg_df = scp[scp['diagnostic'] == 1]

    # Decide source frequency
    has_500 = os.path.isdir(os.path.join(args.data_dir, 'records500'))
    if has_500:
        freq_col, src_freq = 'filename_hr', 500
        print('Using records500 (500 Hz)')
    else:
        freq_col, src_freq = 'filename_lr', 100
        print('records500 not found — falling back to records100 (100 Hz)')

    train_df = df[df['strat_fold'] != args.test_fold]
    test_df  = df[df['strat_fold'] == args.test_fold]

    def process_split(split_df, desc):
        tracings, labels = [], []
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=desc):
            rec_path = os.path.join(args.data_dir, row[freq_col])
            try:
                sig = process_record(rec_path, src_freq)
                tracings.append(sig)
                labels.append(None)  # placeholder
            except Exception as e:
                print(f'  Skipping {rec_path}: {e}')
                labels[-1] = None  # keep in sync

        lbl_matrix = build_labels(split_df.iloc[:len(tracings)], agg_df)
        return np.stack(tracings), lbl_matrix

    print('\nProcessing training records...')
    train_tracings, train_labels = process_split(train_df, 'train')
    write_hdf5(os.path.join(args.output_dir, 'signal_train.h5'), train_tracings, train_labels)

    print('\nProcessing test records...')
    test_tracings, test_labels = process_split(test_df, 'test')
    write_hdf5(os.path.join(args.output_dir, 'signal_test.h5'), test_tracings, test_labels)

    print('\nDone.')
