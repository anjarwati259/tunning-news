import os
import numpy as np
import pandas as pd
import json
from generate_mask import MAR_mask, MNAR_mask_logistic

DATA_DIR = 'datasets'


def split_train_to_new_folder(original_dataname, new_folder_name=None, train_val_split_ratio=0.3, seed=1234):
    """
    Load train.csv dari folder original, split menjadi train dan validation,
    SAVE ke folder BARU (tidak mengubah folder original).
    
    Parameters:
    - original_dataname: Nama folder original (e.g., 'adult')
    - new_folder_name: Nama folder baru yang akan dibuat (default: original_dataname + '_split')
    - train_val_split_ratio: Ratio untuk validation (default: 0.3 = 30%)
    - seed: Random seed untuk reproducibility
    """
    
    if new_folder_name is None:
        new_folder_name = f'{original_dataname}_split'
    
    # Paths
    original_dir = f'{DATA_DIR}/{original_dataname}'
    new_dir = f'{DATA_DIR}/{new_folder_name}'
    original_train_path = f'{original_dir}/train.csv'
    
    print(f'\n{"="*70}')
    print(f'Splitting {original_dataname} to new folder: {new_folder_name}')
    print(f'{"="*70}')
    
    # Create new folder
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        print(f'Created folder: {new_dir}')
    
    # Load original train.csv
    train_df = pd.read_csv(original_train_path)
    total_train = train_df.shape[0]
    
    print(f'\nLoading train.csv from: {original_train_path}')
    print(f'Original train.csv size: {total_train} samples')
    
    # Calculate split sizes
    num_train = int(total_train * (1 - train_val_split_ratio))
    num_val = total_train - num_train
    
    # Create indices dan shuffle
    indices = np.arange(total_train)
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    # Split indices
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    # Create new dataframes
    new_train_df = train_df.iloc[train_indices].reset_index(drop=True)
    val_df = train_df.iloc[val_indices].reset_index(drop=True)
    
    # Save to new folder
    new_train_path = f'{new_dir}/train.csv'
    new_val_path = f'{new_dir}/validation.csv'
    
    new_train_df.to_csv(new_train_path, index=False)
    val_df.to_csv(new_val_path, index=False)
    
    print(f'\nSplitting results:')
    print(f'  New train.csv: {new_train_df.shape[0]} samples ({new_train_df.shape[0]/total_train*100:.1f}%)')
    print(f'  validation.csv: {val_df.shape[0]} samples ({val_df.shape[0]/total_train*100:.1f}%)')
    
    print(f'\nFiles saved to new folder: {new_dir}')
    print(f'  {new_train_path}')
    print(f'  {new_val_path}')
    
    print(f'\n✓ ORIGINAL folder tetap tidak berubah!')
    print(f'  {original_dir}/train.csv (UNCHANGED)')
    print(f'  {original_dir}/test.csv (UNCHANGED)')
    
    print(f'{"="*70}\n')
    
    return new_dir


def generate_masks_for_new_folder(original_dataname, new_folder_name=None, mask_type='MCAR', p=0.3, mask_num=10):
    """
    Generate masks untuk train.csv dan validation.csv di folder BARU.
    
    Mengikuti pattern yang sama dengan generate_mask.py original
    """
    
    if new_folder_name is None:
        new_folder_name = f'{original_dataname}_split'
    
    new_dir = f'{DATA_DIR}/{new_folder_name}'
    original_dir = f'{DATA_DIR}/{original_dataname}'
    
    # Info path - cari di original folder atau di Info subfolder
    info_path_1 = f'{original_dir}/Info/{original_dataname}.json'
    info_path_2 = f'{DATA_DIR}/Info/{original_dataname}.json'
    
    # Coba cari info file
    if os.path.exists(info_path_1):
        info_path = info_path_1
    elif os.path.exists(info_path_2):
        info_path = info_path_2
    else:
        raise FileNotFoundError(f"Info JSON tidak ditemukan di {info_path_1} atau {info_path_2}")
    
    train_path = f'{new_dir}/train.csv'
    val_path = f'{new_dir}/validation.csv'
    
    print(f'\n{"="*70}')
    print(f'Generating {mask_type} masks for {new_folder_name}')
    print(f'{"="*70}')
    
    # Load info
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    
    # Load data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    cols = train_df.columns
    
    # Convert to numpy arrays
    train_num = train_df[cols[num_col_idx]].values.astype(np.float32)
    val_num = val_df[cols[num_col_idx]].values.astype(np.float32)
    
    # Handle categorical columns
    if len(cat_col_idx) > 0:
        train_cat_idx = []
        val_cat_idx = []
        
        cat_columns = train_df[cols[cat_col_idx]].columns
        
        for column in cat_columns:
            # Load mapping JSON untuk column ini
            map_path_idx = f'{original_dir}/{column}_map_idx.json'
            
            with open(map_path_idx, 'r') as f:
                category_to_idx = json.load(f)
            
            # Map kategori string ke index number
            train_cat_idx_i = train_df[column].map(category_to_idx).to_numpy().astype(np.float32)
            val_cat_idx_i = val_df[column].map(category_to_idx).to_numpy().astype(np.float32)
            
            train_cat_idx.append(train_cat_idx_i)
            val_cat_idx.append(val_cat_idx_i)
        
        train_cat = np.stack(train_cat_idx, axis=1)
        val_cat = np.stack(val_cat_idx, axis=1)
        
        train_X = np.concatenate([train_num, train_cat], axis=1)
        val_X = np.concatenate([val_num, val_cat], axis=1)
    
    print(f'Train data shape: {train_X.shape}')
    print(f'Validation data shape: {val_X.shape}')
    print(f'Mask type: {mask_type}, Missing rate: {p*100:.0f}%\n')
    
    # Setup for mask generation (same as generate_mask.py)
    q = 0.3
    if p > 0.3:
        q = 0.1
    
    # Create directory
    folder_name = f'rate{int(p * 100)}'
    mask_dir = f'{new_dir}/masks/{folder_name}/{mask_type}'
    
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    
    # Generate masks (following generate_mask.py pattern)
    for mask_idx in range(mask_num):
        if mask_type == 'MCAR':
            # MCAR: langsung gunakan np.random.rand (tidak ada MCAR_mask function)
            train_mask = np.random.rand(*train_X.shape) < p
            val_mask = np.random.rand(*val_X.shape) < p
            
        elif mask_type == 'MAR':
            # MAR: gunakan MAR_mask function dari generate_mask.py
            train_mask = MAR_mask(train_X, p=p/(1-q), p_obs=q)
            val_mask = MAR_mask(val_X, p=p/(1-q), p_obs=q)
            
        elif mask_type == 'MNAR_logistic_T2':
            # MNAR: gunakan MNAR_mask_logistic function dari generate_mask.py
            train_mask = MNAR_mask_logistic(train_X, p=p, p_params=q, exclude_inputs=True)
            val_mask = MNAR_mask_logistic(val_X, p=p, p_params=q, exclude_inputs=True)
        else:
            raise ValueError('Invalid mask type. Choose from MCAR, MAR, MNAR_logistic_T2')
        
        # Calculate actual missing rates
        train_missing = np.sum(train_mask) / (train_mask.shape[0] * train_mask.shape[1])
        val_missing = np.sum(val_mask) / (val_mask.shape[0] * val_mask.shape[1])
        
        # Save masks
        train_mask_path = f'{mask_dir}/train_mask_{mask_idx}.npy'
        val_mask_path = f'{mask_dir}/val_mask_{mask_idx}.npy'
        
        np.save(train_mask_path, train_mask)
        np.save(val_mask_path, val_mask)
        
        print(f'Mask {mask_idx}: train missing={train_missing:.3f}, val missing={val_missing:.3f}')
    
    print(f'\nMasks saved to: {mask_dir}')
    print(f'{"="*70}\n')


if __name__ == '__main__':
    """
    Contoh penggunaan:
    Split train.csv ke folder baru dan generate masks
    """
    
    # === CONTOH 1: Single Dataset ===
    # print("\n" + "="*70)
    # print("EXAMPLE 1: Single Dataset")
    # print("="*70)
    
    # # Split train.csv dari folder 'adult' ke folder baru 'adult_split'
    # split_train_to_new_folder(
    #     original_dataname='adult',
    #     new_folder_name='adult_split',
    #     train_val_split_ratio=0.3
    # )
    
    # # Generate masks untuk folder baru
    # for mask_type in ['MCAR', 'MAR', 'MNAR_logistic_T2']:
    #     generate_masks_for_new_folder(
    #         original_dataname='adult',
    #         new_folder_name='adult_split',
    #         mask_type=mask_type,
    #         p=0.3,
    #         mask_num=10
    #     )
    
    # === CONTOH 2: Multiple Datasets ===
    # Uncomment untuk jalankan untuk semua dataset
    
    print("\n" + "="*70)
    print("EXAMPLE 2: Multiple Datasets")
    print("="*70)
    
    datasets = ['adult', 'shoppers']
    
    for original_name in datasets:
        new_name = f'{original_name}/{original_name}_validasi'
        
        # Split
        split_train_to_new_folder(original_name, new_name)
        
        # Generate masks
        for mask_type in ['MCAR']:
            generate_masks_for_new_folder(
                original_name, new_name,
                mask_type=mask_type, p=0.3, mask_num=3
            )
    
    print("\nDone!")