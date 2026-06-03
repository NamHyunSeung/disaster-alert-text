"""
재난문자 긴급성 분류 v2 — Level 2 중복 제거 + 키워드 마스킹 Augmentation

변경점 vs train_clean.py:
  - 데이터: dedup_v2 (날짜·숫자 정규화 기준 중복 제거)
  - Augmentation: 학습 시 키워드 30% 확률 마스킹
  - 기본값: epochs=3, batch_size=64 (목표: 1시간 이내)

사용법:
  python train_v2.py
  python train_v2.py --epochs 3 --batch_size 64
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

import torch.nn.functional as F

from dataset_v2 import load_and_split_v2, DisasterDatasetAug
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
    milestones_done = set()

    print(f"[MILESTONE] Epoch {epoch}/{total_epochs} 0% (0/{len(loader)})", flush=True)

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
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', '배치ETA': fmt_time(eta_epoch)})

        pct = step / len(loader)
        for m, label in [(0.25, '25%'), (0.50, '50%'), (0.75, '75%')]:
            if pct >= m and m not in milestones_done:
                print(f"\n[MILESTONE] Epoch {epoch}/{total_epochs} {label} (step {step}/{len(loader)}, loss {avg_loss:.4f})", flush=True)
                milestones_done.add(m)

    return total_loss / len(loader)


def supcon_loss(embeddings, labels, temperature=0.1):
    """Supervised Contrastive Loss on masked text embeddings."""
    embeddings = F.normalize(embeddings, dim=1)
    sim = torch.matmul(embeddings, embeddings.T) / temperature
    # numerical stability
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
    exp_sim = torch.exp(sim)
    # positive mask: same label, exclude diagonal
    lbl = labels.unsqueeze(0)
    pos_mask = torch.eq(lbl, lbl.T).float()
    pos_mask.fill_diagonal_(0)
    # denominator: all pairs except self
    denom_mask = 1 - torch.eye(labels.size(0), device=labels.device)
    log_prob = sim - torch.log((exp_sim * denom_mask).sum(dim=1, keepdim=True) + 1e-8)
    pos_count = pos_mask.sum(dim=1)
    loss_per = -(pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-8)
    valid = pos_count > 0
    return loss_per[valid].mean() if valid.any() else torch.tensor(0.0, device=embeddings.device)


def train_one_epoch_v9(model, loader, optimizer, scheduler, criterion, device,
                       epoch, total_epochs, tokenizer, max_length, alpha, use_ce_masked=True,
                       curriculum_mask=False, alpha_masked=1.0, lambda_con=0.0):
    """CE(원본) + CE(마스킹) + α×KL: masked text에서 직접 정답 지도 + 분포 일관성"""
    from dataset_v2 import _mask_text, _mask_text_partial
    from sklearn.metrics import f1_score, recall_score
    model.train()
    total_loss = 0.0
    epoch_start = time.time()
    milestones_done = set()
    run_preds, run_labels = [], []

    print(f"[MILESTONE] Epoch {epoch}/{total_epochs} 0% (0/{len(loader)})", flush=True)

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train v9]", ncols=100, leave=True)
    for step, batch in enumerate(pbar, 1):
        ids   = batch['input_ids'].to(device)
        attn  = batch['attention_mask'].to(device)
        ttids = batch['token_type_ids'].to(device)
        lbls  = batch['label'].to(device)
        texts = batch['text']

        optimizer.zero_grad()

        # 원본 forward
        logits_orig = model(input_ids=ids, attention_mask=attn, token_type_ids=ttids).logits
        ce_orig = criterion(logits_orig, lbls)

        # 마스킹 텍스트 생성 + 재토크나이징 (curriculum: 배치마다 랜덤 마스킹 비율)
        if curriculum_mask:
            ratio = random.choice([0.3, 0.5, 0.7, 1.0])
            masked_texts = [_mask_text_partial(t, mask_ratio=ratio) if ratio < 1.0 else _mask_text(t) for t in texts]
        else:
            masked_texts = [_mask_text(t) for t in texts]
        enc_m = tokenizer(
            masked_texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt',
        )
        ids_m   = enc_m['input_ids'].to(device)
        attn_m  = enc_m['attention_mask'].to(device)
        ttids_m = enc_m.get('token_type_ids', torch.zeros_like(ids_m)).to(device)

        # 마스킹 forward (SupCon 사용 시 hidden states도 추출)
        out_m = model(input_ids=ids_m, attention_mask=attn_m, token_type_ids=ttids_m,
                      output_hidden_states=(lambda_con > 0))
        logits_masked = out_m.logits

        # CE_masked: masked text에서 정답 레이블로 직접 지도 (핵심)
        ce_masked = criterion(logits_masked, lbls) if use_ce_masked else 0.0

        # KL consistency: masked 예측 분포 → 원본 예측 분포에 맞추기
        p_orig  = F.softmax(logits_orig.detach(), dim=-1)
        log_p_m = F.log_softmax(logits_masked, dim=-1)
        kl_loss = F.kl_div(log_p_m, p_orig, reduction='batchmean')

        # Supervised Contrastive Loss on masked CLS embeddings
        con_loss = torch.tensor(0.0, device=device)
        if lambda_con > 0:
            cls_emb = out_m.hidden_states[-1][:, 0, :]
            con_loss = supcon_loss(cls_emb, lbls)

        loss = ce_orig + alpha_masked * ce_masked + alpha * kl_loss + lambda_con * con_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        avg_loss = total_loss / step

        run_preds.extend(logits_orig.argmax(dim=-1).cpu().numpy())
        run_labels.extend(lbls.cpu().numpy())

        elapsed = time.time() - epoch_start
        eta_epoch = elapsed / step * (len(loader) - step)
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', '배치ETA': fmt_time(eta_epoch)})

        pct = step / len(loader)
        for m, label in [(0.25, '25%'), (0.50, '50%'), (0.75, '75%')]:
            if pct >= m and m not in milestones_done:
                macro_f1 = f1_score(run_labels, run_preds, average='macro', zero_division=0)
                f1_per   = f1_score(run_labels, run_preds, average=None, labels=[0,1,2,3,4], zero_division=0)
                rec_per  = recall_score(run_labels, run_preds, average=None, labels=[0,1,2,3,4], zero_division=0)
                print(
                    f"\n[MILESTONE] Epoch {epoch}/{total_epochs} {label}"
                    f" (step {step}/{len(loader)}, loss {avg_loss:.4f},"
                    f" train macro F1={macro_f1*100:.1f}%)"
                    f"\n  L2 F1={f1_per[2]*100:.1f}%/R={rec_per[2]*100:.1f}%"
                    f"  L3 F1={f1_per[3]*100:.1f}%/R={rec_per[3]*100:.1f}%"
                    f"  L4 F1={f1_per[4]*100:.1f}%/R={rec_per[4]*100:.1f}%",
                    flush=True
                )
                milestones_done.add(m)

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

    acc, macro_f1, _, recall_per, f1_per = compute_metrics(all_labels, all_preds)
    return total_loss / len(loader), acc, macro_f1, f1_per, recall_per


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',       type=str,   default='중요파일/data/raw/재난문자_레이블링결과_dedup_v2.xlsx')
    parser.add_argument('--epochs',     type=int,   default=3)
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--lr',         type=float, default=2e-5)
    parser.add_argument('--max_length', type=int,   default=128)
    parser.add_argument('--mask_prob',  type=float, default=0.3)
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--model_dir',  type=str,   default='model_v2')
    parser.add_argument('--tok_dir',    type=str,   default='tokenizer')
    parser.add_argument('--model_name', type=str,   default='klue/bert-base',
                        help='HuggingFace 모델 이름 (예: monologg/koelectra-small-v3-discriminator)')
    parser.add_argument('--dual_aug',   action='store_true',
                        help='원본+완전마스킹 이중 학습 (Dual Augmentation)')
    parser.add_argument('--asym_aug',   action='store_true',
                        help='비대칭 증강: Label 0/1은 dual(2x), Label 2/3/4는 원본+마스킹×3(4x)')
    parser.add_argument('--masked_ft',  action='store_true',
                        help='마스킹 전용 Fine-tuning: 원본 없음, Label 2/3/4에 부분마스킹 추가')
    parser.add_argument('--init_model', type=str,   default=None,
                        help='Fine-tuning 시작 모델 경로 (기존 학습 모델)')
    parser.add_argument('--v9',        action='store_true',
                        help='KL 일관성 손실: CE(원본)+CE(마스킹)+α×KL(p_원본||p_마스킹)')
    parser.add_argument('--consistency_alpha', type=float, default=1.0,
                        help='KL 일관성 손실 가중치 (기본 1.0)')
    parser.add_argument('--no_ce_masked', action='store_true',
                        help='CE_masked 비활성화 (KL만 사용)')
    parser.add_argument('--save_by_masked_val', action='store_true',
                        help='v9 모드에서 masked val F1 기준으로 best model 저장')
    parser.add_argument('--curriculum_mask', action='store_true',
                        help='curriculum masking: random ratio 30/50/70/100 pct per batch')
    parser.add_argument('--alpha_masked', type=float, default=1.0,
                        help='CE_masked loss weight (default 1.0)')
    parser.add_argument('--lambda_con', type=float, default=0.0,
                        help='Supervised Contrastive loss weight on masked embeddings (default 0.0)')
    parser.add_argument('--save_criterion', type=str, default='l234_min',
                        choices=['macro_f1', 'l234_min'],
                        help='best model 저장 기준: macro_f1 또는 l234_min(L2/L3/L4 F1+Recall 최솟값, 기본)')
    parser.add_argument('--class_weight', type=str, default=None,
                        help='클래스 가중치 (예: "1,1,2,3.5,6"). 지정 시 FocalLoss 대신 CE+weight 사용')
    parser.add_argument('--upsample_synthetic', type=int, default=0,
                        help='is_synthetic=1인 train 샘플 N배 복제 (0=off)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing for FocalLoss (default 0.0)')
    parser.add_argument('--prebuilt', type=str, default=None,
                        help='전처리 완료 parquet 경로 (build_preprocessed.py 출력); 지정 시 on-the-fly 마스킹 생략')
    args = parser.parse_args()

    if args.masked_ft and args.lr == 2e-5:
        args.lr = 5e-6  # fine-tuning 시 낮은 LR
    if args.v9 and args.lr == 2e-5:
        args.lr = 1e-6  # v9 fine-tuning은 더 낮은 LR

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.tok_dir,   exist_ok=True)
    os.makedirs('results',      exist_ok=True)

    print("\n데이터 로드 중...")
    _prebuilt_masked_val_df = None  # prebuilt parquet에서 가져온 masked val DataFrame
    if args.prebuilt:
        _pb = pd.read_parquet(args.prebuilt)
        val_df   = _pb[(_pb['split'] == 'val')  & (_pb['aug_type'] == 'original')][['text', 'label']].reset_index(drop=True)
        test_df  = _pb[(_pb['split'] == 'test') & (_pb['aug_type'] == 'original')][['text', 'label']].reset_index(drop=True)
        train_df = _pb[_pb['split'] == 'train'][['text', 'label']].reset_index(drop=True)
        _prebuilt_masked_val_df = _pb[(_pb['split'] == 'val') & (_pb['aug_type'] == 'masked')][['text', 'label']].reset_index(drop=True)
        print(f"Prebuilt 데이터셋 로드: {args.prebuilt}")
        print(f"  Train(aug포함) {len(train_df):,} / Val {len(val_df):,} / Test {len(test_df):,}")
        print("\n학습 데이터(aug) 레이블 분포:")
        label_info = {0: '긴급 아님', 1: '낮은 긴급성', 2: '중간 긴급성', 3: '높은 긴급성', 4: '매우 높은 긴급성'}
        for lbl, cnt in train_df['label'].value_counts().sort_index().items():
            print(f"  Label {lbl} ({label_info[lbl]}): {cnt:,}건 ({cnt/len(train_df)*100:.1f}%)")
    else:
        train_df, val_df, test_df = load_and_split_v2(args.data, seed=args.seed)
        print(f"Train: {len(train_df):,} / Val: {len(val_df):,} / Test: {len(test_df):,}")
        label_info = {0: '긴급 아님', 1: '낮은 긴급성', 2: '중간 긴급성', 3: '높은 긴급성', 4: '매우 높은 긴급성'}
        print("\n학습 데이터 레이블 분포:")
        for lbl, cnt in train_df['label'].value_counts().sort_index().items():
            print(f"  Label {lbl} ({label_info[lbl]}): {cnt:,}건 ({cnt/len(train_df)*100:.1f}%)")

    print("\n토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.save_pretrained(args.tok_dir)

    val_ds   = DisasterDatasetAug(val_df,   tokenizer, args.max_length, augment=False)
    test_ds  = DisasterDatasetAug(test_df,  tokenizer, args.max_length, augment=False)

    # masked val: save_by_masked_val 시 사용
    masked_val_loader = None
    if (args.v9 or args.masked_ft) and args.save_by_masked_val:
        if _prebuilt_masked_val_df is not None:
            masked_val_ds = DisasterDatasetAug(_prebuilt_masked_val_df, tokenizer, args.max_length, augment=False)
            masked_val_loader = DataLoader(masked_val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
            print("save_by_masked_val: prebuilt masked val 사용")
        else:
            from dataset_v2 import _mask_text
            masked_val_df = val_df.copy()
            masked_val_df['text'] = masked_val_df['text'].apply(_mask_text)
            masked_val_ds = DisasterDatasetAug(masked_val_df, tokenizer, args.max_length, augment=False)
            masked_val_loader = DataLoader(masked_val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
            print("save_by_masked_val: masked val F1 기준으로 best model 저장")

    if args.upsample_synthetic > 0 and 'is_synthetic' in train_df.columns:
        syn = train_df[train_df['is_synthetic'] == 1]
        if len(syn) > 0:
            train_df = pd.concat(
                [train_df] + [syn] * (args.upsample_synthetic - 1), ignore_index=True
            )
            print(f"Synthetic upsampling x{args.upsample_synthetic}: {len(syn)}건 → {len(syn)*args.upsample_synthetic}건 (train 총 {len(train_df):,}건)")

    v9_collate = None  # v9 외에는 default collate 사용

    if args.v9:
        from dataset_v2 import v9_collate_fn
        v9_collate = v9_collate_fn
        train_ds = DisasterDatasetAug(train_df, tokenizer, args.max_length,
                                      augment=False, return_text=True)
        sampler  = make_weighted_sampler(train_df['label'].tolist())
        print(f"v9 Consistency Training: {len(train_df):,}건 원본 데이터 (KL α={args.consistency_alpha})")
    elif args.masked_ft:
        if args.prebuilt:
            # 증강이 사전 계산된 경우: train_df = full_mask + partial_mask_1 + partial_mask_2
            aug_df = train_df
            train_ds = DisasterDatasetAug(aug_df, tokenizer, args.max_length, augment=False)
            sampler  = make_weighted_sampler(aug_df['label'].tolist())
            cnt = aug_df['label'].value_counts().sort_index()
            print(f"Masked FT (prebuilt): {len(aug_df):,}건")
            for lbl in range(5):
                print(f"  Label {lbl}: {cnt.get(lbl, 0):,}건")
        else:
            from dataset_v2 import _mask_text, _mask_text_partial
            hard_labels = {2, 3, 4}
            frames = []
            # 전체 레이블: 완전마스킹 1x (원본 제거)
            mf = train_df.copy(); mf['text'] = mf['text'].apply(_mask_text)
            frames.append(mf)
            # Label 2/3/4: 부분마스킹(60%) 2x 추가
            for _ in range(2):
                mx = train_df[train_df['label'].isin(hard_labels)].copy()
                mx['text'] = mx['text'].apply(lambda t: _mask_text_partial(t, mask_ratio=0.6))
                frames.append(mx)
            aug_df = pd.concat(frames, ignore_index=True)
            train_ds = DisasterDatasetAug(aug_df, tokenizer, args.max_length, augment=False)
            sampler  = make_weighted_sampler(aug_df['label'].tolist())
            cnt = aug_df['label'].value_counts().sort_index()
            print(f"Masked FT: {len(train_df):,} → {len(aug_df):,}건 (완전마스킹 전체 + 부분마스킹 L2/3/4)")
            for lbl in range(5):
                print(f"  Label {lbl}: {cnt.get(lbl, 0):,}건")
    elif args.asym_aug:
        from dataset_v2 import _mask_text
        # Label 0/1: 원본 + 마스킹×1 = 2x
        # Label 2/3/4: 원본 + 마스킹×3 = 4x
        hard_labels = {2, 3, 4}
        frames = [train_df]
        # 전체 1회 마스킹 (Label 0/1도 포함)
        m1 = train_df.copy(); m1['text'] = m1['text'].apply(_mask_text)
        frames.append(m1)
        # Label 2/3/4 추가 마스킹 2회
        for _ in range(2):
            mx = train_df[train_df['label'].isin(hard_labels)].copy()
            mx['text'] = mx['text'].apply(_mask_text)
            frames.append(mx)
        aug_df = pd.concat(frames, ignore_index=True)
        train_ds = DisasterDatasetAug(aug_df, tokenizer, args.max_length, augment=False)
        sampler  = make_weighted_sampler(aug_df['label'].tolist())
        cnt = aug_df['label'].value_counts().sort_index()
        print(f"Asymmetric Aug: 학습 데이터 {len(train_df):,} → {len(aug_df):,}건")
        for lbl in range(5):
            print(f"  Label {lbl}: {cnt.get(lbl, 0):,}건")
    elif args.dual_aug:
        from dataset_v2 import _mask_text
        masked_df = train_df.copy()
        masked_df['text'] = masked_df['text'].apply(_mask_text)
        dual_df = pd.concat([train_df, masked_df], ignore_index=True)
        train_ds = DisasterDatasetAug(dual_df, tokenizer, args.max_length, augment=False)
        sampler  = make_weighted_sampler(dual_df['label'].tolist())
        print(f"Dual Augmentation: 학습 데이터 {len(train_df):,} → {len(dual_df):,}건 (원본+완전마스킹)")
    else:
        train_ds = DisasterDatasetAug(train_df, tokenizer, args.max_length, augment=True, mask_prob=args.mask_prob)
        sampler  = make_weighted_sampler(train_df['label'].tolist())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=0, pin_memory=(device.type == 'cuda'),
                              collate_fn=v9_collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.init_model:
        print(f"\n모델 로드 중 (fine-tuning): {args.init_model}")
        model = load_model(args.init_model, num_labels=5).to(device)
    else:
        print(f"\n모델 로드 중: {args.model_name}")
        model = build_model(num_labels=5, model_name=args.model_name).to(device)
    if args.class_weight:
        import torch.nn as nn
        cw = torch.tensor([float(x) for x in args.class_weight.split(',')], dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.label_smoothing)
        print(f"Criterion: CrossEntropyLoss+weight {args.class_weight}")
    else:
        criterion = FocalLoss(gamma=2.0, label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.10)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\n총 {total_steps:,} 스텝 ({args.epochs} 에폭 × {len(train_loader):,} 배치)")
    print(f"키워드 마스킹 확률: {args.mask_prob*100:.0f}%")
    print("※ GPU 기준 에폭당 약 13~17분 예상\n")

    log_records = []
    best_f1     = 0.0
    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()

        if args.v9:
            train_loss = train_one_epoch_v9(
                model, train_loader, optimizer, scheduler, criterion,
                device, epoch, args.epochs, tokenizer, args.max_length, args.consistency_alpha,
                use_ce_masked=(not args.no_ce_masked),
                curriculum_mask=args.curriculum_mask,
                alpha_masked=args.alpha_masked,
                lambda_con=args.lambda_con,
            )
        else:
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, criterion,
                device, epoch, args.epochs,
            )
        val_loss, val_acc, val_f1, _, _ = eval_one_epoch(
            model, val_loader, criterion, device,
            f"Epoch {epoch}/{args.epochs} [Val]",
        )

        # masked val 평가 (save_by_masked_val 시)
        masked_val_f1 = None
        masked_val_l234_min = None
        if masked_val_loader is not None:
            _, _, masked_val_f1, masked_f1_per, masked_rec_per = eval_one_epoch(
                model, masked_val_loader, criterion, device,
                f"Epoch {epoch}/{args.epochs} [MaskedVal]",
            )
            masked_val_l234_min = min(
                masked_f1_per[2], masked_f1_per[3], masked_f1_per[4],
                masked_rec_per[2], masked_rec_per[3], masked_rec_per[4],
            )

        ep_time       = time.time() - ep_start
        total_elapsed = time.time() - total_start
        eta_total     = (total_elapsed / epoch) * (args.epochs - epoch)

        if masked_val_l234_min is not None:
            masked_str = (
                f" | Masked MacroF1: {masked_val_f1*100:.2f}%"
                f" | L234 min(F1,R): {masked_val_l234_min*100:.2f}%"
                f" [L2 F1={masked_f1_per[2]*100:.1f}%/R={masked_rec_per[2]*100:.1f}%"
                f" L3 F1={masked_f1_per[3]*100:.1f}%/R={masked_rec_per[3]*100:.1f}%"
                f" L4 F1={masked_f1_per[4]*100:.1f}%/R={masked_rec_per[4]*100:.1f}%]"
            )
        else:
            masked_str = ""
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc*100:.2f}% | "
            f"Val Macro F1: {val_f1*100:.2f}%{masked_str} | "
            f"소요: {fmt_time(ep_time)} | "
            f"남은 시간: {fmt_time(eta_total)}"
        )

        log_records.append({
            'epoch': epoch, 'train_loss': train_loss,
            'val_loss': val_loss, 'val_accuracy': val_acc, 'val_macro_f1': val_f1,
            'masked_val_f1': masked_val_f1,
            'masked_val_l234_min': masked_val_l234_min,
        })

        if masked_val_loader is not None:
            if args.save_criterion == 'l234_min':
                save_f1 = masked_val_l234_min
                crit_label = f"L234 min(F1,R): {save_f1*100:.2f}%"
            else:
                save_f1 = masked_val_f1
                crit_label = f"Masked Macro F1: {save_f1*100:.2f}%"
        else:
            save_f1 = val_f1
            crit_label = f"Val Macro F1: {save_f1*100:.2f}%"

        if save_f1 > best_f1:
            best_f1 = save_f1
            model.save_pretrained(args.model_dir)
            print(f"  >> Best model 저장 ({crit_label})")

    tag = args.model_dir.replace('/', '_').replace('\\', '_')
    log_csv     = f'results/train_log_{tag}.csv'
    curves_png  = f'results/training_curves_{tag}.png'
    report_txt  = f'results/evaluation_report_{tag}.txt'
    cm_png      = f'results/confusion_matrix_{tag}.png'

    log_df = pd.DataFrame(log_records)
    log_df.to_csv(log_csv, index=False, encoding='utf-8-sig')
    save_training_curves(log_df, curves_png)
    print(f"\n학습 곡선 저장: {curves_png}")

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
    save_evaluation_report(all_labels, all_preds, report_txt)
    save_confusion_matrix(all_labels, all_preds, cm_png)

    print("\n저장 완료:")
    print(f"  모델:      {args.model_dir}/")
    print(f"  토크나이저: {args.tok_dir}/")
    print(f"  학습 로그:  {log_csv}")
    print(f"  평가 결과:  {report_txt}")
    print(f"  혼동행렬:   {cm_png}")
    print(f"  학습 곡선:  {curves_png}")


if __name__ == '__main__':
    main()
