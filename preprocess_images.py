"""
preprocess_images.py — build image_train.h5 and image_test.h5 from PTB-XL PNGs.

Replicates the data-preparation logic in the original image notebook (Cell 5):
  - globs all *.png files in the flat output directory produced by the generator
  - loads each image as grayscale
  - matches to PTB-XL labels via ecg_id extracted from the filename
  - splits train / test using strat_fold == test_fold
  - saves as HDF5 with keys 'images' (N, H, W uint8) and 'labels' (N, 5)

Usage:
    python preprocess_images.py data/ptbxl data/ptbxl_images \
        --output_dir data/image_hdf5

The generator must have been run first to populate data/ptbxl_images with PNGs.
"""

import argparse
import ast
import glob
import os

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']


def build_labels(df, agg_df):
    def row_label(scp_codes):
        lbl = np.zeros(len(SUPERCLASSES), dtype=np.float32)
        for code in scp_codes.keys():
            if code in agg_df.index:
                sc = agg_df.loc[code, 'diagnostic_class']
                if sc in SUPERCLASSES:
                    lbl[SUPERCLASSES.index(sc)] = 1.0
        return lbl

    return {ecg_id: row_label(row['scp_codes']) for ecg_id, row in df.iterrows()}


def write_hdf5(path, images, labels):
    arr = np.stack(images)
    lbl = np.stack(labels)
    with h5py.File(path, 'w') as f:
        f.create_dataset('images', data=arr, compression='gzip')
        f.create_dataset('labels', data=lbl, compression='gzip')
    print(f'Saved {path}  images={arr.shape}  labels={lbl.shape}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir',   help='PTB-XL root (contains ptbxl_database.csv)')
    parser.add_argument('images_dir', help='flat directory containing all *.png files')
    parser.add_argument('--output_dir', default='data/image_hdf5',
                        help='directory for output HDF5 files (default: data/image_hdf5)')
    parser.add_argument('--test_fold', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load PTB-XL metadata ---
    df = pd.read_csv(os.path.join(args.data_dir, 'ptbxl_database.csv'), index_col='ecg_id')
    df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)

    scp = pd.read_csv(os.path.join(args.data_dir, 'scp_statements.csv'), index_col=0)
    agg_df = scp[scp['diagnostic'] == 1]

    id_to_label = build_labels(df, agg_df)
    test_ids = set(df[df['strat_fold'] == args.test_fold].index)

    # --- Glob all PNGs and match by ecg_id ---
    # Notebook extracts ecg_id from characters 17:22 of the full path string.
    # We extract the numeric prefix of the filename stem instead (e.g. "00001_lr" → 1).
    png_paths = sorted(glob.glob(os.path.join(args.images_dir, '**', '*.png'), recursive=True))
    if not png_paths:
        png_paths = sorted(glob.glob(os.path.join(args.images_dir, '*.png')))

    if not png_paths:
        raise FileNotFoundError(
            f'No PNG files found in {args.images_dir}. '
            'Run the image generator first.'
        )
    print(f'Found {len(png_paths)} PNG files')

    train_images, train_labels = [], []
    test_images,  test_labels  = [], []
    skipped = 0

    for p in tqdm(png_paths, desc='Loading PNGs'):
        stem = os.path.splitext(os.path.basename(p))[0]  # e.g. "00001_lr"
        try:
            ecg_id = int(stem.split('_')[0])
        except ValueError:
            skipped += 1
            continue

        if ecg_id not in id_to_label:
            skipped += 1
            continue

        img = np.array(Image.open(p).convert('L'), dtype=np.uint8)
        lbl = id_to_label[ecg_id]

        if ecg_id in test_ids:
            test_images.append(img)
            test_labels.append(lbl)
        else:
            train_images.append(img)
            train_labels.append(lbl)

    if skipped:
        print(f'Skipped {skipped} files (unrecognised filename format or missing ecg_id)')

    print(f'Train: {len(train_images)} images | Test: {len(test_images)} images')

    write_hdf5(os.path.join(args.output_dir, 'image_train.h5'), train_images, train_labels)
    write_hdf5(os.path.join(args.output_dir, 'image_test.h5'),  test_images,  test_labels)

    print('Done.')
