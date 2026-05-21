"""
Step 5: 2단계 파이프라인
  Stage 1 — 규칙 기반 분류 (고확신 케이스 즉시 판정)
  Stage 2 — KF-DeBERTa 파인튜닝 (규칙이 위임한 케이스 처리)

평가: 규칙 커버리지 / Stage2 단독 / 파이프라인 통합
"""

import os, json, time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
MODEL_NAME   = "kakaobank/kf-deberta-base"
MAX_LENGTH   = 128
MAX_EPOCHS   = 7
LR           = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
FOCAL_GAMMA  = 2.0
ES_PATIENCE  = 2
NUM_LABELS   = 3
LABEL_NAMES  = ['긴급', '주의', '일반']
SAVE_DIR     = "kfdeberta_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"장치: {device}")
if device.type == "cuda":
    BATCH_SIZE  = 32
    MAX_SAMPLES = None
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    BATCH_SIZE  = 16
    MAX_SAMPLES = 10_000
    print("[경고] CPU 모드: 10,000개 샘플로 실험")


# ─────────────────────────────────────────
# Stage 1: 규칙 기반 분류기
# ─────────────────────────────────────────
# 고확신 키워드만 사용 (오분류 위험 최소화)
_CERT_EMERG = [
    '즉시 대피', '대피명령', '대피 명령', '긴급대피', '긴급 대피', '신속히 대피',
    '지진 발생', '쓰나미', '민방공 경보', '민방공경보', '테러 발생',
]
_CERT_CAUTION_TYPES   = ['호우경보', '호우주의보', '태풍경보', '태풍주의보',
                          '한파경보', '한파주의보', '폭염경보', '폭염주의보',
                          '대설경보', '대설주의보', '강풍경보', '강풍주의보',
                          '풍랑경보', '풍랑주의보']
_CERT_GENERAL_MISSING = ['찾습니다', '실종된']     # 실종자 수색


def rule_classify(msg: str):
    """
    확신할 수 있는 케이스만 판정, 나머지는 None 반환(모델 위임).
    False Positive를 줄이기 위해 보수적으로 설계.
    """
    # 긴급: 즉각 생명 위협 키워드
    if any(kw in msg for kw in _CERT_EMERG):
        return 0

    # 주의: 재해 유형 + 경보/주의보가 붙어 있는 복합 패턴
    if any(pat in msg for pat in _CERT_CAUTION_TYPES):
        return 1

    # 일반: 실종자 수색 (person info + URL 패턴)
    if any(kw in msg for kw in _CERT_GENERAL_MISSING) and 'cm' in msg:
        return 2

    return None   # 모델에 위임


# ─────────────────────────────────────────
# Focal Loss / Early Stopping / 과적합 모니터
# ─────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce  = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt  = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class EarlyStopping:
    def __init__(self, patience=2):
        self.patience = patience
        self.counter  = 0
        self.best     = None

    def step(self, score) -> bool:
        if self.best is None or score > self.best:
            self.best    = score
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class OverfittingMonitor:
    GAP_WARN    = 0.10
    GAP_OVERFIT = 0.20
    UNDERFIT    = 0.40

    def __init__(self):
        self.history = []

    def update(self, epoch, tr_loss, val_loss, val_f1) -> str:
        gap = val_loss - tr_loss
        self.history.append(dict(epoch=epoch, tr=tr_loss, val=val_loss,
                                  gap=gap, f1=val_f1))
        if tr_loss > self.UNDERFIT:
            return '과소적합'
        if gap > self.GAP_OVERFIT:
            return '과적합'
        if gap > self.GAP_WARN and len(self.history) >= 2 \
                and val_loss > self.history[-2]['val']:
            return '과적합 주의'
        return '정상'


# ─────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────
class DisasterDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.enc    = tokenizer(texts.tolist(), truncation=True,
                                padding='max_length', max_length=max_len,
                                return_tensors='pt')
        self.labels = torch.tensor(labels.tolist(), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.enc.items()} | \
               {'labels': self.labels[idx]}


# ─────────────────────────────────────────
# 평가 함수
# ─────────────────────────────────────────
def evaluate(model, loader, criterion):
    model.eval()
    total, preds, labels = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            y = batch.pop('labels').to(device)
            x = {k: v.to(device) for k, v in batch.items()}
            out = model(**x)
            total += criterion(out.logits, y).item()
            preds.extend(out.logits.argmax(-1).cpu().numpy())
            labels.extend(y.cpu().numpy())
    n = len(loader)
    return (total / n,
            accuracy_score(labels, preds),
            f1_score(labels, preds, average='macro',    zero_division=0),
            f1_score(labels, preds, average='weighted', zero_division=0),
            preds, labels)


# ─────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────
print("\n데이터 로드 중...")
train_df = pd.read_csv('data_train.csv', encoding='utf-8-sig').fillna({'메시지내용': ''})
val_df   = pd.read_csv('data_val.csv',   encoding='utf-8-sig').fillna({'메시지내용': ''})
test_df  = pd.read_csv('data_test.csv',  encoding='utf-8-sig').fillna({'메시지내용': ''})
print(f"Train {len(train_df):,} / Val {len(val_df):,} / Test {len(test_df):,}")


def stratified_sample(df, n_per_class, seed=42):
    parts = [df[df['label'] == lbl].sample(
        min(len(df[df['label'] == lbl]), n_per_class), random_state=seed)
        for lbl in range(NUM_LABELS)]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


if MAX_SAMPLES:
    train_df = stratified_sample(train_df, MAX_SAMPLES // NUM_LABELS)
    val_df   = stratified_sample(val_df,   max(200, MAX_SAMPLES // 10) // NUM_LABELS)
    print(f"샘플링 후 -> Train {len(train_df):,} / Val {len(val_df):,}")

print(f"Train 분포:\n{train_df['label_name'].value_counts().to_string()}\n")

# 클래스 가중치
lc = train_df['label'].value_counts().sort_index()
class_weights = torch.tensor(
    [len(train_df) / (NUM_LABELS * lc[i]) for i in range(NUM_LABELS)],
    dtype=torch.float,
).to(device)
print(f"클래스 가중치: {class_weights.cpu().numpy().round(3)}")


# ─────────────────────────────────────────
# 토크나이저 & 데이터셋
# ─────────────────────────────────────────
print(f"\n토크나이저 로드: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("토크나이징 중...")
train_ds = DisasterDataset(train_df['메시지내용'], train_df['label'], tokenizer, MAX_LENGTH)
val_ds   = DisasterDataset(val_df['메시지내용'],   val_df['label'],   tokenizer, MAX_LENGTH)
test_ds  = DisasterDataset(test_df['메시지내용'],  test_df['label'],  tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=0,
                          pin_memory=(device.type == 'cuda'))
val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=0)


# ─────────────────────────────────────────
# 모델 / 옵티마이저 / 스케줄러
# ─────────────────────────────────────────
print(f"\n모델 로드: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS)
model.to(device)

criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

total_steps  = len(train_loader) * MAX_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler    = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
print(f"총 스텝 {total_steps:,} / 워밍업 {warmup_steps:,}")


# ─────────────────────────────────────────
# 학습 루프
# ─────────────────────────────────────────
history    = {k: [] for k in ['train_loss', 'val_loss', 'val_acc', 'val_f1', 'gap']}
es         = EarlyStopping(ES_PATIENCE)
of_monitor = OverfittingMonitor()
best_f1    = 0.0
best_epoch = 0
stop_reason = "최대 에폭"

print(f"\n학습 시작 (max {MAX_EPOCHS} epochs, patience={ES_PATIENCE})")
print("=" * 60)
t_start = time.time()

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    ep_loss, ep_n = 0.0, 0
    ep_t = time.time()

    for step, batch in enumerate(train_loader, 1):
        y = batch.pop('labels').to(device)
        x = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        loss = criterion(model(**x).logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        ep_loss += loss.item(); ep_n += 1

        if step % 100 == 0:
            print(f"  E{epoch} {step:>5}/{len(train_loader)} "
                  f"loss={ep_loss/ep_n:.4f} ({time.time()-ep_t:.0f}s)")

    tr_loss  = ep_loss / ep_n
    vl, va, vf, vwf, _, _ = evaluate(model, val_loader, criterion)
    gap    = vl - tr_loss
    status = of_monitor.update(epoch, tr_loss, vl, vf)

    history['train_loss'].append(tr_loss)
    history['val_loss'].append(vl)
    history['val_acc'].append(va)
    history['val_f1'].append(vf)
    history['gap'].append(gap)

    print(f"\nEpoch {epoch}/{MAX_EPOCHS}  ({time.time()-ep_t:.0f}s)")
    print(f"  Train {tr_loss:.4f}  Val {vl:.4f}  Acc {va:.4f}  F1 {vf:.4f}")
    print(f"  Gap {gap:+.4f}  [{status}]")

    if vf > best_f1:
        best_f1, best_epoch = vf, epoch
        os.makedirs(SAVE_DIR, exist_ok=True)
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        print(f"  [Best] 저장 (Val F1: {best_f1:.4f})")

    if es.step(vf):
        stop_reason = f"Early Stopping (patience={ES_PATIENCE})"
        print(f"\n  {stop_reason}")
        break
    print()

total_time   = time.time() - t_start
actual_ep    = len(history['train_loss'])
print(f"\n학습 완료: {total_time/60:.1f}분 | Best Epoch {best_epoch} | {stop_reason}")


# ─────────────────────────────────────────
# Test 평가 (Best 모델)
# ─────────────────────────────────────────
print("\nBest 모델 로드 및 Test 평가...")
model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR)
model.to(device)

tl, ta, tf, twf, t_preds, t_labels = evaluate(model, test_loader, criterion)
print(f"\n=== Test 결과 (Stage 2 단독) ===")
print(f"  Accuracy  {ta:.4f}  Macro F1  {tf:.4f}  Weighted F1  {twf:.4f}")
print("\n" + classification_report(t_labels, t_preds,
                                   target_names=LABEL_NAMES, zero_division=0))


# ─────────────────────────────────────────
# 파이프라인 평가 (Stage1 + Stage2)
# ─────────────────────────────────────────
print("파이프라인 통합 평가 중...")

# Stage 1 커버리지 분석
test_msgs    = test_df['메시지내용'].tolist()
rule_results = [rule_classify(m) for m in test_msgs]
rule_covered = sum(r is not None for r in rule_results)
rule_pct     = rule_covered / len(test_msgs) * 100

print(f"\n[Stage 1] 규칙 커버: {rule_covered:,}/{len(test_msgs):,} ({rule_pct:.1f}%)")

# 규칙 커버 케이스 정확도
rule_true = [t_labels[i] for i, r in enumerate(rule_results) if r is not None]
rule_pred = [r             for i, r in enumerate(rule_results) if r is not None]
if rule_pred:
    rule_acc = accuracy_score(rule_true, rule_pred)
    rule_f1  = f1_score(rule_true, rule_pred, average='macro', zero_division=0)
    print(f"[Stage 1] 정확도 {rule_acc:.4f}  Macro F1 {rule_f1:.4f}")

# 파이프라인 최종 예측
pipe_preds = []
for i, r in enumerate(rule_results):
    pipe_preds.append(r if r is not None else t_preds[i])

pipe_acc = accuracy_score(t_labels, pipe_preds)
pipe_f1  = f1_score(t_labels, pipe_preds, average='macro',    zero_division=0)
pipe_wf1 = f1_score(t_labels, pipe_preds, average='weighted', zero_division=0)

print(f"\n=== 파이프라인 최종 결과 ===")
print(f"  Accuracy  {pipe_acc:.4f}  Macro F1  {pipe_f1:.4f}  Weighted F1  {pipe_wf1:.4f}")
print("\n" + classification_report(t_labels, pipe_preds,
                                   target_names=LABEL_NAMES, zero_division=0))


# ─────────────────────────────────────────
# 시각화 1: 과적합 진단
# ─────────────────────────────────────────
ep_x = range(1, actual_ep + 1)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('KF-DeBERTa 학습 과정 및 과적합 진단', fontsize=13)

axes[0].plot(ep_x, history['train_loss'], 'b-o', label='Train')
axes[0].plot(ep_x, history['val_loss'],   'r-o', label='Val')
axes[0].fill_between(ep_x, history['train_loss'], history['val_loss'],
                     alpha=0.15, color='orange')
axes[0].axvline(best_epoch, color='green', linestyle='--', label=f'Best E{best_epoch}')
axes[0].set_title('Loss')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(ep_x, history['val_f1'], 'm-o')
axes[1].axvline(best_epoch, color='green', linestyle='--')
axes[1].set_title('Val Macro F1')
axes[1].set_ylim(0, 1); axes[1].grid(alpha=0.3)

gc = ['green' if g < OverfittingMonitor.GAP_WARN
      else ('orange' if g < OverfittingMonitor.GAP_OVERFIT else 'red')
      for g in history['gap']]
axes[2].bar(ep_x, history['gap'], color=gc)
axes[2].axhline(OverfittingMonitor.GAP_WARN,    color='orange', linestyle='--',
                label=f'주의({OverfittingMonitor.GAP_WARN})')
axes[2].axhline(OverfittingMonitor.GAP_OVERFIT, color='red',    linestyle='--',
                label=f'과적합({OverfittingMonitor.GAP_OVERFIT})')
axes[2].set_title('Gap (Val-Train Loss)')
axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('kfdeberta_diagnosis.png', dpi=150, bbox_inches='tight')
print("\n과적합 진단 저장: kfdeberta_diagnosis.png")


# ─────────────────────────────────────────
# 시각화 2: 혼동 행렬 (파이프라인)
# ─────────────────────────────────────────
cm      = confusion_matrix(t_labels, pipe_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('파이프라인 혼동 행렬 (Test)', fontsize=13)
for ax, data, fmt, title in zip(
    axes, [cm, cm_norm], ['d', '.2f'], ['절대값', '비율']
):
    sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_xlabel('예측'); ax.set_ylabel('실제'); ax.set_title(title)
plt.tight_layout()
plt.savefig('kfdeberta_confusion.png', dpi=150, bbox_inches='tight')
print("혼동 행렬 저장: kfdeberta_confusion.png")


# ─────────────────────────────────────────
# 전체 모델 비교
# ─────────────────────────────────────────
try:
    results_df = pd.read_csv('results.csv', encoding='utf-8-sig')
except FileNotFoundError:
    results_df = pd.DataFrame()

new_row = pd.DataFrame([{
    'model':             f'KF-DeBERTa Pipeline (E{best_epoch})',
    'val_acc':           max(history['val_acc']),
    'val_macro_f1':      best_f1,
    'test_acc':          pipe_acc,
    'test_macro_f1':     pipe_f1,
    'test_weighted_f1':  pipe_wf1,
    'train_time_sec':    round(total_time, 1),
}])
results_df = pd.concat([results_df, new_row], ignore_index=True)
results_df.to_csv('results.csv', index=False, encoding='utf-8-sig')

print("\n=== 전체 모델 비교 ===")
print(results_df[['model', 'test_acc', 'test_macro_f1',
                   'test_weighted_f1']].to_string(index=False))

# 비교 시각화
if len(results_df) >= 2:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(results_df['model'], results_df['test_macro_f1'],
            color='#e74c3c', edgecolor='black', linewidth=0.5)
    ax.set_xlim(0.7, 1.0)
    ax.set_title('Macro F1 비교 (Test)')
    ax.set_xlabel('Macro F1')
    for i, v in enumerate(results_df['test_macro_f1']):
        ax.text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig('all_models_comparison.png', dpi=150, bbox_inches='tight')
    print("비교 저장: all_models_comparison.png")


# ─────────────────────────────────────────
# 요약 저장
# ─────────────────────────────────────────
max_gap   = max(history['gap'])
of_status = ('과적합' if max_gap > OverfittingMonitor.GAP_OVERFIT
             else ('과적합 주의' if max_gap > OverfittingMonitor.GAP_WARN
                   else '정상'))

summary = f"""
=== KF-DeBERTa + 규칙 파이프라인 학습 요약 ===
모델     : {MODEL_NAME}
손실함수 : Focal Loss (gamma={FOCAL_GAMMA})
총 에폭  : {actual_ep}/{MAX_EPOCHS}  ({stop_reason})
Best 에폭: {best_epoch}  (Val Macro F1: {best_f1:.4f})
학습 시간: {total_time/60:.1f}분

=== Stage 1 (규칙 기반) ===
커버리지 : {rule_pct:.1f}%  ({rule_covered:,}/{len(test_msgs):,}건)
정확도   : {rule_acc:.4f}  Macro F1: {rule_f1:.4f}

=== Stage 2 (KF-DeBERTa 단독) ===
Accuracy : {ta:.4f}
Macro F1 : {tf:.4f}

=== 파이프라인 통합 ===
Accuracy : {pipe_acc:.4f}
Macro F1 : {pipe_f1:.4f}

=== 과적합 진단: {of_status} (최대 gap={max_gap:.4f}) ===
{'에폭':>4}  {'Train':>7}  {'Val':>7}  {'Gap':>7}  {'F1':>7}
{'─'*44}"""
for h in of_monitor.history:
    m = " <- Best" if h['epoch'] == best_epoch else ""
    summary += f"\n{h['epoch']:>4}  {h['tr']:>7.4f}  {h['val']:>7.4f}  {h['gap']:>+7.4f}  {h['f1']:>7.4f}{m}"

with open('kfdeberta_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)
print("\n요약 저장: kfdeberta_summary.txt")
print("\n=== Step 5 완료 ===")
