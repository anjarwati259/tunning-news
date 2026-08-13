"""
Grid Search Tuning: alpha & beta — Full Pipeline
=================================================
Loss formula: loss = alpha * class_loss + beta * recon_loss

Untuk setiap kombinasi alpha & beta, jalankan FULL pipeline:
  1. Train embedding dengan alpha/beta tersebut
  2. Normalisasi embedding
  3. Train diffusion (DiffPutter) — identik dengan main_mrmd.py
  4. Imputasi in-sample & out-of-sample
  5. Evaluasi ACC, MAE, RMSE hasil imputasi

Pilih alpha/beta dengan hasil imputasi OUT-SAMPLE terbaik berdasarkan --metric.

OUTPUT:
  - CSV utama (FORMAT TIDAK DIUBAH, sama seperti sebelumnya) → out-sample,
    dipakai untuk ranking & pemilihan best config:
      tuning_results/tuning_{dataname}_{mask}_{ratio}_{split_idx}_{metric}.csv
  - CSV baru (terpisah) → in-sample saja, hanya untuk referensi/pelaporan:
      tuning_results/tuning_{dataname}_{mask}_{ratio}_{split_idx}_{metric}_insample.csv

RESUME SUPPORT:
Jika file CSV utama (out-sample) sudah ada dan berisi kombinasi (alpha, beta)
yang sudah pernah dijalankan, kombinasi tersebut akan di-SKIP otomatis dan
baris lama tetap dipakai untuk ranking. Aman dijalankan ulang untuk
melanjutkan dari kombinasi yang belum selesai, tanpa mengulang dari awal.

Grid coarse: alpha & beta ∈ [0.0, 0.25, 0.5, 0.75, 1.0] (5×5 = 25 kombinasi)

Cara pakai:
    python tune_alpha_beta.py \
        --dataname shoppers \
        --split_idx 0 \
        --ratio 30 \
        --mask MCAR \
        --gpu 0 \
        --metric acc
"""

import os
import shutil
import csv
import time
import argparse
import warnings
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ── Argparse ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Grid Search Tuning alpha-beta Full Pipeline')
parser.add_argument('--dataname',   type=str,   default='shoppers')
parser.add_argument('--split_idx',  type=int,   default=0)
parser.add_argument('--ratio',      type=str,   default='30')
parser.add_argument('--mask',       type=str,   default='MCAR')
parser.add_argument('--gpu',        type=int,   default=0)
parser.add_argument('--metric',     type=str,   default='acc',
                    help='Metric penentu ranking (berdasarkan OUT-SAMPLE): acc (maximize) | mae | rmse (minimize)')
parser.add_argument('--noise_std',  type=float, default=0.00,  help='Noise std embedding (sama dengan main_mrmd.py)')
parser.add_argument('--hid_dim',    type=int,   default=1024,  help='Hidden dim diffusion (sama dengan main_mrmd.py)')
parser.add_argument('--epochs',     type=int,   default=10000, help='Epoch diffusion training')
parser.add_argument('--max_iter',   type=int,   default=6,     help='Jumlah iterasi EM')
parser.add_argument('--num_trials', type=int,   default=10,    help='Jumlah sampling imputasi')
parser.add_argument('--num_steps',  type=int,   default=50,    help='Jumlah diffusion steps')
args = parser.parse_args()

assert args.metric in ('acc', 'mae', 'rmse'), \
    f"--metric harus salah satu dari: acc, mae, rmse. Diberikan: {args.metric}"

if not torch.cuda.is_available():
    raise RuntimeError('GPU tidak tersedia!')

device    = f'cuda:{args.gpu}'
mask_type = 'MNAR_logistic_T2' if args.mask == 'MNAR' else args.mask

# ── Grid Coarse ───────────────────────────────────────────────────────────────
COARSE_ALPHA = [0.0]
COARSE_BETA  = [0.0, 0.25, 0.5, 0.75, 1.0]

# ── Import dari kode asli ─────────────────────────────────────────────────────
import sys
sys.path.insert(0, '.')
from dataset_mrmd import (
    load_dataset, get_eval, mean_std,
    SupervisedLearnableEmbeddingModel,
    encode_with_embedding,
)
from model import MLPDiffusion, Model
from diffusion_utils import sample_step, impute_mask

print(f'\n{"="*60}')
print(f'Grid Search Tuning — Full Pipeline')
print(f'Dataset    : {args.dataname} | {mask_type} | ratio={args.ratio} | split={args.split_idx}')
print(f'Metric     : {args.metric} (berdasarkan out-sample)')
print(f'Pipeline   : epochs={args.epochs} | max_iter={args.max_iter} | '
      f'num_trials={args.num_trials} | num_steps={args.num_steps}')
print(f'Grid coarse: alpha={COARSE_ALPHA}')
print(f'             beta ={COARSE_BETA}')
print(f'Total coarse: {len(COARSE_ALPHA)*len(COARSE_BETA)} kombinasi')
print(f'{"="*60}\n')

# ── Load dataset satu kali (MRmD + komponen dasar) ───────────────────────────
print('[INFO] Loading dataset komponen dasar (MRmD discretization)...')
# load_dataset sudah handle internal untuk load train/test dari folder split
(train_X_default, test_X_default,
 ori_train_mask, ori_test_mask,
 train_num, test_num,
 train_all_idx, test_all_idx,
 extend_train_mask, extend_test_mask,
 cat_bin_num,
 emb_model_default,
 emb_sizes,
 mrmd,
 bin_midpoints,
 n_num_cols,
 t_mrmd, t_emb_default
) = load_dataset(args.dataname, args.split_idx, mask_type, args.ratio, args.noise_std)

# ── Rekonstruksi komponen embedding dari load_dataset() ──────────────────────
import pandas as pd, json
from sklearn.preprocessing import LabelEncoder
from dataset_mrmd import compute_embedding_size

info_path = f'datasets/Info/{args.dataname}.json'
with open(info_path, 'r') as f:
    info = json.load(f)

num_col_idx    = info['num_col_idx']
cat_col_idx    = info['cat_col_idx']
target_col_idx = info['target_col_idx']

data_df  = pd.read_csv(f'datasets/{args.dataname}/data.csv')
train_df = pd.read_csv(f'datasets/{args.dataname}/{args.dataname}_validasi/train.csv')
test_df  = pd.read_csv(f'datasets/{args.dataname}/{args.dataname}_validasi/validation.csv')
cols     = train_df.columns

train_y       = train_df[cols[target_col_idx]]
test_y        = test_df[cols[target_col_idx]]  # validation targets
label_encoder = LabelEncoder()
label_encoder.fit(pd.concat([train_y, test_y]).values.ravel().astype(str))
train_labels  = label_encoder.transform(train_y.values.ravel().astype(str))
n_classes     = len(label_encoder.classes_)

num_bins   = mrmd.n_bins_ if mrmd is not None else []
cat_dims_cat = []
if len(cat_col_idx) > 0:
    cat_columns = cols[cat_col_idx]
    data_cat    = data_df[cat_columns].astype(str)
    for col in cat_columns:
        le = LabelEncoder()
        le.fit(data_cat[col])
        cat_dims_cat.append(len(le.classes_))

all_dims      = num_bins + cat_dims_cat
cat_idx_array = train_all_idx

torch.cuda.set_device(args.gpu)

print(f'[INFO] cat_idx_array: {cat_idx_array.shape}')
print(f'[INFO] all_dims     : {all_dims}')
print(f'[INFO] emb_sizes    : {emb_sizes}')
print(f'[INFO] n_classes    : {n_classes}')
print(f'[INFO] n_num_cols   : {n_num_cols}')


# ── Helper: safe float & format ───────────────────────────────────────────────
def _safe_float(v):
    try:
        return float(v)
    except:
        return float('nan')

def _fmt(v, d=6):
    f = _safe_float(v)
    return round(f, d) if not np.isnan(f) else float('nan')

def _prt(v, d=4):
    f = _safe_float(v)
    return f'{f:.{d}f}' if not np.isnan(f) else 'nan'


# ── Train embedding dengan alpha & beta custom ────────────────────────────────
def train_embedding(alpha, beta):
    """Identik dengan train_supervised_embedding_model() di dataset_mrmd.py."""
    if alpha == 0.0 and beta == 0.0:
        print(f'  [SKIP] alpha=0 & beta=0 → skip.')
        return None

    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)

    model = SupervisedLearnableEmbeddingModel(
        all_dims, emb_sizes, n_classes,
        dropout=0.1, hidden_dim=256,
        use_mlp=True, mlp_ratio=1.5,
        noise_std=args.noise_std,
    ).to(device)

    optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce_loss    = nn.CrossEntropyLoss()
    cat_tensor = torch.tensor(cat_idx_array, dtype=torch.long, device=device)
    lbl_tensor = torch.tensor(train_labels,  dtype=torch.long, device=device)
    loader     = DataLoader(
        torch.utils.data.TensorDataset(cat_tensor, lbl_tensor),
        batch_size=1024, shuffle=True,
        num_workers=0, generator=torch.Generator(device='cpu'),
    )

    best_loss, patience_counter, best_state = float('inf'), 0, None

    model.train()
    for epoch in range(300):
        total_loss = total_c = total_r = 0.0
        nb = 0
        for bc, bl in loader:
            optimizer.zero_grad()
            z, cl, rl = model(bc, add_noise=True)
            c_loss = ce_loss(cl, bl) if alpha > 0 \
                     else torch.tensor(0.0, device=device)
            r_loss = sum(ce_loss(rl[i], bc[:, i])
                         for i in range(model.n_cols)) / model.n_cols \
                     if beta > 0 else torch.tensor(0.0, device=device)
            loss = alpha * c_loss + beta * r_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_c    += c_loss.item()
            total_r    += r_loss.item()
            nb         += 1

        avg = total_loss / nb
        if (epoch + 1) % 10 == 0:
            print(f'    [Emb] Epoch {epoch+1} loss={avg:.4f} '
                  f'(class={total_c/nb:.4f}, recon={total_r/nb:.4f})')

        if avg < best_loss:
            best_loss, patience_counter = avg, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= 30:
            print(f'    [Emb] Early stopping epoch {epoch+1}')
            break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ── Full pipeline: embedding → diffusion → imputasi → evaluasi ───────────────
def run_full_pipeline(alpha, beta):
    """
    Jalankan full pipeline identik dengan main_mrmd.py.
    Kembalikan:
      mae_out, rmse_out, acc_out   (out-of-sample, iterasi terakhir → dipakai ranking)
      mae_in,  rmse_in,  acc_in    (in-sample, iterasi terakhir → disimpan ke CSV terpisah)
      t_emb, ckpt_dir
    """
    print(f'\n  [EMB] Training embedding alpha={alpha}, beta={beta}...')
    t_emb_start = time.time()
    emb_model   = train_embedding(alpha, beta)
    t_emb       = time.time() - t_emb_start

    if emb_model is None:
        return (float('nan'), float('nan'), float('nan'),
                float('nan'), float('nan'), float('nan'),
                t_emb, None)

    train_X = encode_with_embedding(emb_model, cat_idx_array,       device)
    test_X  = encode_with_embedding(emb_model, test_all_idx,        device)

    torch.set_default_device(device)

    mean_X, std_X = mean_std(train_X, extend_train_mask)
    std_X[std_X == 0] = 1.0
    in_dim = train_X.shape[1]

    X      = torch.tensor((train_X - mean_X) / std_X / 2,
                          device=device, dtype=torch.float32)
    X_test = torch.tensor((test_X  - mean_X) / std_X / 2,
                          device=device, dtype=torch.float32)

    mask_train = torch.tensor(extend_train_mask, device=device, dtype=torch.float32)
    mask_test  = torch.tensor(extend_test_mask,  device=device, dtype=torch.float32)

    mean_X_gpu = torch.tensor(mean_X, device=device, dtype=torch.float32)
    std_X_gpu  = torch.tensor(std_X,  device=device, dtype=torch.float32)

    std_np  = std_X_gpu.cpu().numpy()
    mean_np = mean_X_gpu.cpu().numpy()

    hid_dim    = args.hid_dim
    num_trials = args.num_trials
    num_steps  = args.num_steps
    len_num    = 0

    ckpt_dir = (f'ckpt_tuning/{args.dataname}/rate{args.ratio}/{mask_type}/'
                f'{args.split_idx}/current')

    MAEs_out, RMSEs_out, ACCs_out = [], [], []
    MAEs_in,  RMSEs_in,  ACCs_in  = [], [], []
    start_time = time.time()

    for iteration in range(args.max_iter):
        print(f'\n  [ITER {iteration}] alpha={alpha}, beta={beta}')
        os.makedirs(f'{ckpt_dir}/{iteration}', exist_ok=True)

        # ── M-Step: Train diffusion ──────────────────────────────────────────
        if iteration == 0:
            X_miss     = (1. - mask_train) * X
            train_data = X_miss
        else:
            rec_prev   = torch.tensor(
                np.load(f'{ckpt_dir}/iter_{iteration}.npy') / 2,
                device=device, dtype=torch.float32
            )
            X_miss     = rec_prev * mask_train + X * (1. - mask_train)
            train_data = X_miss

        class GPUTensorDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]

        train_loader = DataLoader(
            GPUTensorDataset(train_data),
            batch_size=4096, shuffle=True,
            num_workers=0,
            generator=torch.Generator(device=device),
        )

        num_epochs = args.epochs + 1
        denoise_fn = MLPDiffusion(in_dim, hid_dim).to(device)
        model_diff = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)
        optimizer  = torch.optim.Adam(model_diff.parameters(), lr=1e-4, weight_decay=0)
        scheduler  = ReduceLROnPlateau(optimizer, mode='min', factor=0.9,
                                       patience=50)

        model_diff.train()
        best_loss_diff = float('inf')
        patience_diff  = 0

        pbar = tqdm(range(num_epochs),
                    desc=f'  Diffusion iter={iteration} a={alpha} b={beta}')
        for epoch in pbar:
            batch_loss = 0.0
            len_input  = 0
            for batch in train_loader:
                inputs     = batch.float()
                loss       = model_diff(inputs).mean()
                batch_loss += loss.item() * len(inputs)
                len_input  += len(inputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            curr_loss = batch_loss / len_input
            scheduler.step(curr_loss)

            if curr_loss < best_loss_diff:
                best_loss_diff = curr_loss
                patience_diff  = 0
                torch.save(model_diff.state_dict(),
                           f'{ckpt_dir}/{iteration}/model.pt')
            else:
                patience_diff += 1
                if patience_diff == 500:
                    print('  Early stopping diffusion')
                    break

            pbar.set_postfix(loss=curr_loss)
            if epoch % 1000 == 0:
                torch.save(model_diff.state_dict(),
                           f'{ckpt_dir}/{iteration}/model_{epoch}.pt')

        end_time = time.time()

        # ── E-Step: In-sample Imputation ────────────────────────────────────
        rec_Xs = []
        for trial in tqdm(range(num_trials), desc='  In-sample imputation'):
            X_miss_in = (1. - mask_train) * X
            denoise_fn = MLPDiffusion(in_dim, hid_dim).to(device)
            model_diff = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)
            model_diff.load_state_dict(
                torch.load(f'{ckpt_dir}/{iteration}/model.pt'))
            net = model_diff.denoise_fn_D
            rec_X = impute_mask(net, X_miss_in, mask_train,
                                X.shape[0], X.shape[1], num_steps, device)
            rec_X = rec_X * mask_train.float() + X * (1. - mask_train.float())
            rec_X = torch.clamp(rec_X, -10.0, 10.0)
            rec_Xs.append(rec_X)

        rec_X     = torch.stack(rec_Xs, dim=0).mean(0)
        np.save(f'{ckpt_dir}/iter_{iteration+1}.npy',
                (rec_X * 2).cpu().numpy())

        pred_X = rec_X.cpu().numpy() * 2 * std_np + mean_np
        X_true = X.cpu().numpy()     * 2 * std_np + mean_np

        # In-sample evaluation — disimpan ke CSV terpisah, TIDAK dipakai ranking
        mae_in, rmse_in, acc_in = get_eval(
            dataname=args.dataname, X_recon=pred_X, X_true=X_true,
            truth_all_idx=train_all_idx, num_num=len_num,
            emb_model=emb_model, emb_sizes=emb_sizes,
            mask=ori_train_mask, device=device, oos=False,
            bin_midpoints=bin_midpoints, n_num_cols=n_num_cols,
            num_true_norm=train_num,
        )
        MAEs_in.append(mae_in)
        RMSEs_in.append(rmse_in)
        ACCs_in.append(acc_in)
        print(f'  In-sample  → MAE={_prt(mae_in)}  RMSE={_prt(rmse_in)}  ACC={_prt(acc_in)}')

        # ── E-Step: Out-of-sample Imputation ────────────────────────────────
        rec_Xs_out = []
        for trial in tqdm(range(num_trials), desc='  Out-of-sample imputation'):
            X_miss_out = (1. - mask_test) * X_test
            denoise_fn = MLPDiffusion(in_dim, hid_dim).to(device)
            model_diff = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)
            model_diff.load_state_dict(
                torch.load(f'{ckpt_dir}/{iteration}/model.pt'))
            net   = model_diff.denoise_fn_D
            rec_X = impute_mask(net, X_miss_out, mask_test,
                                X_test.shape[0], X_test.shape[1], num_steps, device)
            rec_X = rec_X * mask_test.float() + X_test * (1. - mask_test.float())
            rec_X = torch.clamp(rec_X, -10.0, 10.0)
            rec_Xs_out.append(rec_X)

        rec_X_out = torch.stack(rec_Xs_out, dim=0).mean(0)

        pred_X_out = rec_X_out.cpu().numpy() * 2 * std_np + mean_np
        X_true_out = X_test.cpu().numpy()    * 2 * std_np + mean_np

        # Out-of-sample evaluation — DIPAKAI UNTUK RANKING, masuk CSV utama (format lama)
        mae_out, rmse_out, acc_out = get_eval(
            dataname=args.dataname, X_recon=pred_X_out, X_true=X_true_out,
            truth_all_idx=test_all_idx, num_num=len_num,
            emb_model=emb_model, emb_sizes=emb_sizes,
            mask=ori_test_mask, device=device, oos=True,
            bin_midpoints=bin_midpoints, n_num_cols=n_num_cols,
            num_true_norm=test_num,
        )
        MAEs_out.append(mae_out)
        RMSEs_out.append(rmse_out)
        ACCs_out.append(acc_out)

        print(f'  Out-sample → MAE={_prt(mae_out)}  RMSE={_prt(rmse_out)}  ACC={_prt(acc_out)}')
        start_time = time.time()

    torch.set_default_device('cpu')

    final_mae_out  = MAEs_out[-1]  if MAEs_out  else float('nan')
    final_rmse_out = RMSEs_out[-1] if RMSEs_out else float('nan')
    final_acc_out  = ACCs_out[-1]  if ACCs_out  else float('nan')

    final_mae_in   = MAEs_in[-1]  if MAEs_in  else float('nan')
    final_rmse_in  = RMSEs_in[-1] if RMSEs_in else float('nan')
    final_acc_in   = ACCs_in[-1]  if ACCs_in  else float('nan')

    return (final_mae_out, final_rmse_out, final_acc_out,
            final_mae_in,  final_rmse_in,  final_acc_in,
            t_emb, ckpt_dir)


# ── Fungsi score (selalu berdasarkan OUT-SAMPLE, format CSV utama tidak berubah) ──
def score(row):
    if args.metric == 'acc':
        v = _safe_float(row['acc'])
        return -v if not np.isnan(v) else float('inf')
    elif args.metric == 'mae':
        v = _safe_float(row['mae'])
        return v if not np.isnan(v) else float('inf')
    else:
        v = _safe_float(row['rmse'])
        return v if not np.isnan(v) else float('inf')


# ── Setup output ──────────────────────────────────────────────────────────────
os.makedirs('tuning_results', exist_ok=True)

# CSV UTAMA — format SAMA seperti sebelumnya, out-sample, dipakai untuk ranking
out_csv = (f'tuning_results/tuning_{args.dataname}_{args.mask}_'
           f'{args.ratio}_{args.split_idx}_{args.metric}.csv')
fieldnames = ['phase', 'alpha', 'beta', 'acc', 'mae', 'rmse', 'time_s']

# CSV BARU — terpisah, khusus in-sample, hanya untuk referensi
out_csv_insample = (f'tuning_results/tuning_{args.dataname}_{args.mask}_'
                     f'{args.ratio}_{args.split_idx}_{args.metric}_insample.csv')
fieldnames_insample = ['phase', 'alpha', 'beta', 'acc', 'mae', 'rmse', 'time_s']

# ── RESUME: baca kombinasi yang sudah pernah dijalankan (dari CSV utama) ─────
coarse_results = []
done_combos    = set()

if os.path.exists(out_csv):
    print(f'[RESUME] File CSV utama ditemukan: {out_csv}')
    with open(out_csv, 'r', newline='') as f_csv:
        reader = csv.DictReader(f_csv)
        for row in reader:
            coarse_results.append(row)
            done_combos.add((round(_safe_float(row['alpha']), 4),
                              round(_safe_float(row['beta']),  4)))
    print(f'[RESUME] {len(done_combos)} kombinasi sudah selesai, akan di-SKIP:')
    for a, b in sorted(done_combos):
        print(f'    alpha={a}, beta={b}')
else:
    print('[INFO] Tidak ada file CSV utama sebelumnya, mulai dari awal.')

# ── Phase 1: Coarse Search ────────────────────────────────────────────────────
total_coarse = len(COARSE_ALPHA) * len(COARSE_BETA)

print(f'\n{"─"*60}')
print(f'[PHASE 1] Coarse Search — {total_coarse} kombinasi total '
      f'({len(done_combos)} sudah selesai, {total_coarse - len(done_combos)} tersisa)')
print(f'{"─"*60}')

# CSV utama: tulis header hanya jika belum ada
if not os.path.exists(out_csv):
    with open(out_csv, 'w', newline='') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

# CSV in-sample: tulis header hanya jika belum ada
if not os.path.exists(out_csv_insample):
    with open(out_csv_insample, 'w', newline='') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames_insample)
        writer.writeheader()

with open(out_csv, 'a', newline='') as f_csv_main, \
     open(out_csv_insample, 'a', newline='') as f_csv_in:

    writer_main = csv.DictWriter(f_csv_main, fieldnames=fieldnames)
    writer_in   = csv.DictWriter(f_csv_in,   fieldnames=fieldnames_insample)

    for i, (alpha, beta) in enumerate(itertools.product(COARSE_ALPHA, COARSE_BETA), 1):
        combo_key = (round(alpha, 4), round(beta, 4))

        if combo_key in done_combos:
            print(f'\n[{i:02d}/{total_coarse}] alpha={alpha:.2f}  beta={beta:.2f}  → SKIP (sudah selesai)')
            continue

        print(f'\n[{i:02d}/{total_coarse}] alpha={alpha:.2f}  beta={beta:.2f}')

        t0 = time.time()
        (mae, rmse, acc,
         mae_in, rmse_in, acc_in,
         t_emb, ckpt_dir) = run_full_pipeline(alpha, beta)
        elapsed = time.time() - t0

        # Baris CSV utama — format SAMA seperti sebelumnya (out-sample)
        row_main = {
            'phase':   'coarse',
            'alpha':   alpha,
            'beta':    beta,
            'acc':     _fmt(acc),
            'mae':     _fmt(mae),
            'rmse':    _fmt(rmse),
            'time_s':  round(elapsed, 2),
        }
        writer_main.writerow(row_main)
        f_csv_main.flush()
        coarse_results.append(row_main)
        done_combos.add(combo_key)

        # Baris CSV in-sample — file terpisah
        row_in = {
            'phase':   'coarse',
            'alpha':   alpha,
            'beta':    beta,
            'acc':     _fmt(acc_in),
            'mae':     _fmt(mae_in),
            'rmse':    _fmt(rmse_in),
            'time_s':  round(elapsed, 2),
        }
        writer_in.writerow(row_in)
        f_csv_in.flush()

        print(f'  → HASIL OUT-SAMPLE : ACC={_prt(acc)}  MAE={_prt(mae)}  RMSE={_prt(rmse)}  (→ {out_csv})')
        print(f'  → HASIL IN-SAMPLE  : ACC={_prt(acc_in)}  MAE={_prt(mae_in)}  RMSE={_prt(rmse_in)}  (→ {out_csv_insample})')
        print(f'  → total={elapsed/3600:.2f}jam')

        if ckpt_dir is not None and os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)
            print(f"  [CKPT] Dihapus: {ckpt_dir}")
        torch.cuda.empty_cache()

# Ranking coarse (berdasarkan out-sample, sesuai --metric) — TIDAK BERUBAH
coarse_sorted = sorted(coarse_results, key=score)
print(f'\n[COARSE] Ranking berdasarkan {args.metric} (out-sample):')
for rank, r in enumerate(coarse_sorted[:10], 1):
    print(f'  #{rank}: alpha={r["alpha"]}, beta={r["beta"]} '
          f'→ ACC={_prt(r["acc"])}  MAE={_prt(r["mae"])}  RMSE={_prt(r["rmse"])}')


# ── Pilih Best Config (berdasarkan out-sample) — TIDAK BERUBAH ──────────────
all_results = coarse_results
best        = min(all_results, key=score)

print(f'\n{"="*60}')
print(f'TUNING SELESAI')
print(f'  Dataset           : {args.dataname}')
print(f'  Metric optimisasi : {args.metric}')
print(f'  Best alpha        : {best["alpha"]}')
print(f'  Best beta         : {best["beta"]}')
print(f'  ACC               : {_prt(best["acc"])}')
print(f'  MAE               : {_prt(best["mae"])}')
print(f'  RMSE              : {_prt(best["rmse"])}')
print(f'{"="*60}')

# Simpan best config — TIDAK BERUBAH
best_txt = (f'tuning_results/best_config_{args.dataname}_{args.mask}_'
            f'{args.ratio}_{args.split_idx}_{args.metric}.txt')
with open(best_txt, 'w') as f:
    f.write(f'# Best alpha-beta — Full Pipeline Grid Search Tuning\n')
    f.write(f'# Dataset  : {args.dataname}\n')
    f.write(f'# Mask     : {args.mask} (ratio={args.ratio}, split={args.split_idx})\n')
    f.write(f'# Metric   : {args.metric}\n\n')
    f.write(f'alpha = {best["alpha"]}\n')
    f.write(f'beta  = {best["beta"]}\n\n')
    f.write(f'# Hasil imputasi out-of-sample iterasi terakhir:\n')
    f.write(f'ACC  = {_prt(best["acc"])}\n')
    f.write(f'MAE  = {_prt(best["mae"])}\n')
    f.write(f'RMSE = {_prt(best["rmse"])}\n\n')
    f.write(f'# Untuk dipakai di dataset_mrmd.py, ubah baris 498-499 menjadi:\n')
    f.write(f'    alpha = {best["alpha"]}\n')
    f.write(f'    beta  = {best["beta"]}\n')

print(f'\nBest config disimpan di      : {best_txt}')
print(f'Hasil out-sample disimpan di  : {out_csv}')
print(f'Hasil in-sample disimpan di   : {out_csv_insample}')

# ── Ranking Akhir Top-10 ──────────────────────────────────────────────────────
all_sorted = sorted(all_results, key=score)
print(f'\nTop-10 Konfigurasi (metric={args.metric}):')
print(f'  {"#":>3}  {"Phase":7}  {"alpha":>6}  {"beta":>6}  '
      f'{"ACC":>8}  {"MAE":>8}  {"RMSE":>8}  {"Jam":>6}')
print('  ' + '─'*65)
for rank, r in enumerate(all_sorted[:10], 1):
    jam = _safe_float(r['time_s']) / 3600
    print(f'  {rank:>3}  {r["phase"]:7}  {float(r["alpha"]):>6.2f}  {float(r["beta"]):>6.2f}  '
          f'{_prt(r["acc"]):>8}  {_prt(r["mae"]):>8}  {_prt(r["rmse"]):>8}  {jam:>5.2f}j')