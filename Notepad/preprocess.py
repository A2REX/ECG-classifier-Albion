"""
preprocess.py — Generate all H5 data files before running the notebooks.

Run from the Notepad/ directory:
    python preprocess.py --dataset L   # full dataset (~19k records)
    python preprocess.py --dataset S   # small dataset (2000 signal / 500 image)

Outputs (in Notepad/data/):
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
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

PTB_PATH = '../ptb-xl-dataset/'
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

    os.makedirs('data', exist_ok=True)

    if dataset == 'S':
        h5_test, h5_train   = 'data/signal_test_s.h5',  'data/signal_train_s.h5'
        txt_test, txt_train = 'Test_RECORDS_LowRes_s.txt', 'Train_RECORDS_LowRes_s.txt'
        data_size = 2000
    else:
        h5_test, h5_train   = 'data/signal_test_l.h5',  'data/signal_train_l.h5'
        txt_test, txt_train = 'Test_RECORDS_LowRes.txt', 'Train_RECORDS_LowRes.txt'
        data_size = None  # all records

    Y = load_ptbxl_labels()

    # Read full record list (generate RECORDS_LowRes.txt from RECORDS if missing)
    lowres_path = PTB_PATH + 'RECORDS_LowRes.txt'
    if not os.path.exists(lowres_path):
        print('  RECORDS_LowRes.txt not found — generating from RECORDS...')
        with open(PTB_PATH + 'RECORDS', encoding='utf-8') as f:
            all_records = [l.rstrip('\n') for l in f]
        lowres = [r for r in all_records if r.endswith('_lr')]
        with open(lowres_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lowres))
        print(f'  Written: {lowres_path} ({len(lowres)} records)')
    with open(lowres_path, encoding='utf-8') as f:
        lines = [l.rstrip('\n') for l in f]

    # Split records by fold (look up fold via ecg_id extracted from path)
    lines_test, lines_train = [], []
    for line in lines:
        ecg_id = int(os.path.basename(line).split('_')[0])
        if Y.loc[ecg_id, 'strat_fold'] == TEST_FOLD:
            lines_test.append(line)
        else:
            lines_train.append(line)

    def lines_to_labels(line_list):
        ecg_ids = [int(os.path.basename(l).split('_')[0]) for l in line_list]
        return to_binary(Y.loc[ecg_ids].diagnostic_superclass.tolist())

    y_test  = lines_to_labels(lines_test)
    y_train = lines_to_labels(lines_train)

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

    os.makedirs('data', exist_ok=True)

    if dataset == 'S':
        h5_test, h5_train = 'data/test_s.h5', 'data/train_s.h5'
        data_size = 500
    else:
        h5_test, h5_train = 'data/test_l.h5', 'data/train_l.h5'
        data_size = None  # all images

    if skip(h5_test) and skip(h5_train):
        return

    Y = load_ptbxl_labels()
    y_test_series  = Y[Y.strat_fold == TEST_FOLD].diagnostic_superclass
    y_train_series = Y[Y.strat_fold != TEST_FOLD].diagnostic_superclass

    png_paths = sorted(glob.glob(PNG_FOLDER + '*.png'))
    if not png_paths:
        print(f'ERROR: No PNG files found in {PNG_FOLDER}')
        sys.exit(1)

    # Build a lookup from ecg_id -> PNG path (first match per ECG)
    png_by_id = {}
    for p in png_paths:
        eid = int(os.path.basename(p).split('_')[0])
        if eid not in png_by_id:
            png_by_id[eid] = p

    # Use the same ECG ID order as the signal txt files to guarantee alignment
    if dataset == 'S':
        sig_txt_test  = PTB_PATH + 'Test_RECORDS_LowRes_s.txt'
        sig_txt_train = PTB_PATH + 'Train_RECORDS_LowRes_s.txt'
    else:
        sig_txt_test  = PTB_PATH + 'Test_RECORDS_LowRes.txt'
        sig_txt_train = PTB_PATH + 'Train_RECORDS_LowRes.txt'

    def ecg_ids_from_txt(txt_path):
        with open(txt_path, encoding='utf-8') as f:
            return [int(os.path.basename(l.rstrip('\n')).split('_')[0]) for l in f if l.strip()]

    test_ecg_ids  = ecg_ids_from_txt(sig_txt_test)
    train_ecg_ids = ecg_ids_from_txt(sig_txt_train)

    png_test  = [png_by_id[eid] for eid in test_ecg_ids  if eid in png_by_id]
    png_train = [png_by_id[eid] for eid in train_ecg_ids if eid in png_by_id]

    if data_size is not None:
        ratio           = len(png_test) / (len(png_test) + len(png_train))
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
        for p in tqdm(png_list, desc='Loading images'):
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
