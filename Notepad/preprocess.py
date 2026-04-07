"""
preprocess.py — Generate all H5 data files before running the notebooks.

Run from the Notepad/ directory:
    python preprocess.py --dataset L   # full dataset (~19k records)
    python preprocess.py --dataset S   # small dataset (2000 signal / 500 image)

Outputs (in Notepad/):
    signal_train_l.h5 / signal_train_s.h5
    signal_test_l.h5  / signal_test_s.h5
    train_l.h5        / train_s.h5
    test_l.h5         / test_s.h5
"""

import os
import sys
import ast
import glob
import argparse
import subprocess

# Guard: ensure we're running inside the NoteEnv conda environment
try:
    import xmljson  # required by ecg-preprocessing-main
except ModuleNotFoundError:
    print("ERROR: 'xmljson' not found — this script must be run with the NoteEnv conda environment.")
    print("  conda activate NoteEnv")
    print("  python preprocess.py --dataset S")
    sys.exit(1)

import numpy as np
import pandas as pd
import h5py
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────

PTB_PATH = (r'..\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
            r'\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3' + '\\')
PNG_FOLDER = r'..\output-images' + '\\'
CLASSES    = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
TEST_FOLD  = 10

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_ptbxl_labels():
    Y = pd.read_csv(PTB_PATH + 'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)
    agg_df = pd.read_csv(PTB_PATH + 'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        return list(set(
            agg_df.loc[k].diagnostic_class
            for k in y_dic if k in agg_df.index
        ))

    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)
    return Y


def to_binary(label_list):
    out = []
    for entry in label_list:
        vec = np.zeros(len(CLASSES))
        for z, cls in enumerate(CLASSES):
            if cls in entry:
                vec[z] = 1
        out.append(vec)
    return np.stack(out)


def skip(path):
    if os.path.exists(path):
        print(f'  SKIP (exists): {path}')
        return True
    return False


# ── Signal H5 generation ──────────────────────────────────────────────────────

def generate_signal_h5(dataset):
    print('\n=== Signal H5 generation ===')

    if dataset == 'S':
        h5_test, h5_train   = 'signal_test_s.h5',  'signal_train_s.h5'
        txt_test, txt_train = 'Test_RECORDS_LowRes_s.txt', 'Train_RECORDS_LowRes_s.txt'
        data_size = 2000
    else:
        h5_test, h5_train   = 'signal_test_l.h5',  'signal_train_l.h5'
        txt_test, txt_train = 'Test_RECORDS_LowRes.txt', 'Train_RECORDS_LowRes.txt'
        data_size = None  # all records

    Y = load_ptbxl_labels()

    # Read full record list
    with open(PTB_PATH + 'RECORDS_LowRes.txt', encoding='utf-8') as f:
        lines = [l.rstrip('\n') for l in f]

    # Split records and labels by fold
    lines_test, lines_train = [], []
    for i, line in enumerate(lines):
        if Y.strat_fold.iloc[i] == TEST_FOLD:
            lines_test.append(line)
        else:
            lines_train.append(line)

    y_test  = to_binary(Y[Y.strat_fold == TEST_FOLD].diagnostic_superclass.tolist())
    y_train = to_binary(Y[Y.strat_fold != TEST_FOLD].diagnostic_superclass.tolist())

    if data_size is not None:
        ratio           = len(y_test) / len(lines)
        data_size_test  = int(ratio * data_size)
        data_size_train = data_size - data_size_test
        lines_test  = lines_test[:data_size_test]
        lines_train = lines_train[:data_size_train]
        y_test  = y_test[:data_size_test]
        y_train = y_train[:data_size_train]

    # Write record list files
    txt_test_path  = PTB_PATH + txt_test
    txt_train_path = PTB_PATH + txt_train

    if not os.path.exists(txt_test_path):
        with open(txt_test_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines_test))
        print(f'  Written: {txt_test_path}')

    if not os.path.exists(txt_train_path):
        with open(txt_train_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines_train))
        print(f'  Written: {txt_train_path}')

    # Generate signal H5 via ecg-preprocessing-main
    preprocess_cmd = [
        sys.executable,
        'ecg-preprocessing-main/generate_h5.py',
        '--new_freq', '400',
        '--new_len', '4096',
        '--remove_baseline',
        '--use_all_leads',
    ]

    if not skip(h5_test):
        print(f'  Generating {h5_test} ...')
        subprocess.run(preprocess_cmd + [txt_test_path, h5_test], check=True)

    with h5py.File(h5_test, 'r+') as f:
        if 'labels' not in f:
            f.create_dataset('labels', data=y_test, compression='gzip')
            print(f'  Labels added to {h5_test}')

    if not skip(h5_train):
        print(f'  Generating {h5_train} ...')
        subprocess.run(preprocess_cmd + [txt_train_path, h5_train], check=True)

    with h5py.File(h5_train, 'r+') as f:
        if 'labels' not in f:
            f.create_dataset('labels', data=y_train, compression='gzip')
            print(f'  Labels added to {h5_train}')

    print('Signal H5 done.')


# ── Image H5 generation ───────────────────────────────────────────────────────

def generate_image_h5(dataset):
    print('\n=== Image H5 generation ===')

    if dataset == 'S':
        h5_test, h5_train = 'test_s.h5', 'train_s.h5'
        data_size = 500
    else:
        h5_test, h5_train = 'test_l.h5', 'train_l.h5'
        data_size = None  # all images

    if skip(h5_test) and skip(h5_train):
        return

    Y = load_ptbxl_labels()

    png_paths = sorted(glob.glob(PNG_FOLDER + '*.png'))
    if not png_paths:
        print(f'ERROR: No PNG files found in {PNG_FOLDER}')
        sys.exit(1)

    y_test_series  = Y[Y.strat_fold == TEST_FOLD].diagnostic_superclass
    y_train_series = Y[Y.strat_fold != TEST_FOLD].diagnostic_superclass

    png_test  = [p for p in png_paths
                 if int(os.path.basename(p).split('_')[0]) in y_test_series.index]
    png_train = [p for p in png_paths
                 if int(os.path.basename(p).split('_')[0]) in y_train_series.index]

    if data_size is not None:
        ratio           = len(png_test) / len(png_paths)
        data_size_test  = int(ratio * data_size)
        data_size_train = data_size - data_size_test
        png_test  = png_test[:data_size_test]
        png_train = png_train[:data_size_train]

    # Build label arrays aligned to PNG order
    def build_labels(png_list, y_series):
        ecg_ids = [int(os.path.basename(p).split('_')[0]) for p in png_list]
        label_list = y_series.loc[ecg_ids].tolist()
        return to_binary(label_list)

    def load_images(png_list):
        arrays = []
        for p in png_list:
            img = Image.open(p).convert('L')
            arrays.append(np.array(img))
        arr = np.stack(arrays).astype(np.float32) / 255.0
        H = arr.shape[1]
        return arr[:, int(0.275 * H):, :]  # crop top 27.5%

    if not skip(h5_test):
        print(f'  Loading {len(png_test)} test images...')
        arr_test = load_images(png_test)
        y_test   = build_labels(png_test, y_test_series)
        with h5py.File(h5_test, 'w') as f:
            f.create_dataset('images', data=arr_test, compression='gzip')
            f.create_dataset('labels', data=y_test,   compression='gzip')
        print(f'  Saved {h5_test}: images {arr_test.shape}, labels {y_test.shape}')

    if not skip(h5_train):
        print(f'  Loading {len(png_train)} train images...')
        arr_train = load_images(png_train)
        y_train   = build_labels(png_train, y_train_series)
        with h5py.File(h5_train, 'w') as f:
            f.create_dataset('images', data=arr_train, compression='gzip')
            f.create_dataset('labels', data=y_train,   compression='gzip')
        print(f'  Saved {h5_train}: images {arr_train.shape}, labels {y_train.shape}')

    print('Image H5 done.')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate H5 data files for ECG notebooks.')
    parser.add_argument('--dataset', choices=['L', 'S'], default='S',
                        help='L = full dataset, S = small subset (default: S)')
    args = parser.parse_args()

    print(f'Dataset: {"large" if args.dataset == "L" else "small"}')
    generate_signal_h5(args.dataset)
    generate_image_h5(args.dataset)
    print('\nAll done. You can now run the notebooks.')
