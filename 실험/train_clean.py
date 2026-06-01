"""
재난문자 긴급성 분류 — 중복 제거 데이터 재학습

사용법:
  python train_clean.py
  python train_clean.py --epochs 5 --batch_size 32
"""

import os
import time
import random
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from collections import Counter
from tqdm import tqdm

from dataset_clean import load_and_split_clean, DisasterDataset
from model import build_model, load_model
from loss import FocalLoss
from utils import (
    compute_metrics, print_evaluation,
    save_evaluation_report, save_confusion_matrix, save_training_curves,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def make_weighted_sampler(labels):
    counts = Counter(labels)
    n = len(labels)
    weights = [n / counts[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=n, replacement=True)


def fmt_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    epoch_start = time.time()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", ncols=100, leave=True)
    for step, batch in enumerate(pbar, 1):
        ids   = batch['input_ids'].to(device)
        mask  = batch['attention_mask'].to(device)
        ttids = batch['token_type_ids'].to(device)
        lbls  = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=ids, attention_mask=mask, token_type_ids=ttids).logits
        loss = criterion(logits, lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        avg_loss = total_loss / step

        elapsed = time.time() - epoch_start
        eta_epoch = elapsed / step * (len(loader) - step)
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            '배치ETA': fmt_time(eta_epoch),
        })

    return total_loss / len(loader)


def eval_one_epoch(model, loader, criterion, device, desc: str):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, ncols=100, leave=False):
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            ttids = batch['token_type_ids'].to(device)
            lbls  = batch['label'].to(device)

            logits = model(input_ids=ids, attention_mask=mask, token_type_ids=ttids).logits
            loss = criterion(logits, lbls)
            total_loss += loss.item()

            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbls.cpu().numpy())

    acc, macro_f1, *_ = compute_metrics(all_labels, all_preds)
    return total_loss / len(loader), acc, macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',       type=str,   default='중요파일/data/raw/재난문자_레이블링결과_dedup.xlsx')
    parser.add_argument('--epochs',     type=int,   default=5)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=2e-5)
    parser.add_argument('--max_length', type=int,   default=128)
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--model_dir',  type=str,   default='model_clean')
    parser.add_argument('--tok_dir',    type=str,   default='tokenizer')
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.tok_dir,   exist_ok=True)
    os.makedirs('results',      exist_ok=True)

    print("\n데이터 로드 중...")
    train_df, val_df, test_df = load_and_split_clean(args.data, seed=args.seed)
    print(f"Train: {len(train_df):,} / Val: {len(val_df):,} / Test: {len(test_df):,}")

    label_counts = train_df['label'].value_counts().sort_index()
    label_info = {
        0: '긴급 아님',  1: '낮은 긴급성', 2: '중간 긴급성',
        3: '높은 긴급성', 4: '매우 높은 긴급성',
    }
    print("\n학습 데이터 레이블 분포:")
    for lbl, cnt in label_counts.items():
        print(f"  Label {lbl} ({label_info[lbl]}): {cnt:,}건 ({cnt/len(train_df)*100:.1f}%)")

    print("\n토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
    tokenizer.save_pretrained(args.tok_dir)

    train_ds = DisasterDataset(train_df, tokenizer, args.max_length)
    val_ds   = DisasterDataset(val_df,   tokenizer, args.max_length)
    test_ds  = DisasterDataset(test_df,  tokenizer, args.max_length)

    sampler = make_weighted_sampler(train_df['label'].tolist())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,  num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("\nKLUE-BERT 모델 로드 중...")
    model = build_model(num_labels=5).to(device)
    criterion = FocalLoss(gamma=2.0)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.10)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\n총 {total_steps:,} 스텝 ({args.epochs} 에폭 × {len(train_loader):,} 배치)")
    print("※ GPU 기준 에폭당 약 3~8분, CPU 기준 약 20~40분 예상\n")

    log_records = []
    best_f1     = 0.0
    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, epoch, args.epochs,
        )
        val_loss, val_acc, val_f1 = eval_one_epoch(
            model, val_loader, criterion, device,
            f"Epoch {epoch}/{args.epochs} [Val]",
        )

        ep_time       = time.time() - ep_start
        total_elapsed = time.time() - total_start
        eta_total     = (total_elapsed / epoch) * (args.epochs - epoch)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc*100:.2f}% | "
            f"Val Macro F1: {val_f1*100:.2f}% | "
            f"소요: {fmt_time(ep_time)} | "
            f"남은 시간: {fmt_time(eta_total)}"
        )

        log_records.append({
            'epoch':        epoch,
            'train_loss':   train_loss,
            'val_loss':     val_loss,
            'val_accuracy': val_acc,
            'val_macro_f1': val_f1,
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained(args.model_dir)
            print(f"  >> Best model 저장 (Val Macro F1: {best_f1*100:.2f}%)")

    log_df = pd.DataFrame(log_records)
    log_df.to_csv('results/train_log_clean.csv', index=False, encoding='utf-8-sig')
    save_training_curves(log_df, 'results/training_curves_clean.png')
    print("\n학습 곡선 저장: results/training_curves_clean.png")

    total_time = time.time() - total_start
    print(f"\n총 학습 시간: {fmt_time(total_time)}")
    print("\nBest model 로드 후 Test 평가 중...")

    best_model = load_model(args.model_dir, num_labels=5).to(device)
    best_model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Test]", ncols=100):
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            ttids = batch['token_type_ids'].to(device)
            lbls  = batch['label']

            logits = best_model(input_ids=ids, attention_mask=mask, token_type_ids=ttids).logits
            preds  = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbls.numpy())

    print_evaluation(all_labels, all_preds)
    save_evaluation_report(all_labels, all_preds, 'results/evaluation_report_clean.txt')
    save_confusion_matrix(all_labels, all_preds, 'results/confusion_matrix_clean.png')

    print("\n저장 완료:")
    print(f"  모델:      {args.model_dir}/")
    print(f"  토크나이저: {args.tok_dir}/")
    print(f"  학습 로그:  results/train_log_clean.csv")
    print(f"  평가 결과:  results/evaluation_report_clean.txt")
    print(f"  혼동행렬:   results/confusion_matrix_clean.png")
    print(f"  학습 곡선:  results/training_curves_clean.png")


if __name__ == '__main__':
    main()
