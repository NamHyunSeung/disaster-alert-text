"""
Step 4: KoELECTRA-v3 + Focal Loss

모델  : monologg/koelectra-base-v3-discriminator
손실함수: Focal Loss (gamma=2) - 불균형 클래스(긴급 1.3%) 집중 학습
추가기능: Early Stopping / 과적합 모니터링 / 전체 모델 비교
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
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
MODEL_NAME   = "monologg/koelectra-base-v3-discriminator"
MAX_LENGTH   = 128
MAX_EPOCHS   = 7          # Early Stopping으로 조기 종료 가능
LR           = 3e-5       # ELECTRA 권장 LR (BERT보다 약간 높게)
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
FOCAL_GAMMA  = 2.0        # Focal Loss 집중도 (0=CE, 2=권장)
ES_PATIENCE  = 2          # Early Stopping patience (에폭 단위)
NUM_LABELS   = 3
LABEL_NAMES  = ['긴급', '주의', '일반']
SAVE_DIR     = "koelectra_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"장치: {device}")
if device.type == "cuda":
    BATCH_SIZE  = 32
    MAX_SAMPLES = None
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    BATCH_SIZE  = 16
    MAX_SAMPLES = 10_000
    print("[경고] GPU 없음 -> CPU 모드: 10,000개 샘플로 실험")

# ──────────────────────────────────────────────
# Focal Loss
# ──────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -(1 - p_t)^gamma * log(p_t)
    - gamma=0 이면 일반 CrossEntropy와 동일
    - gamma 클수록 어려운 샘플(긴급)에 집중, 쉬운 샘플(일반) 가중치 감소
    """
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight  # 클래스 불균형 보정 가중치

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt      = torch.exp(-ce_loss)                    # 정답 클래스 확률
        focal   = (1.0 - pt) ** self.gamma * ce_loss     # 쉬운 샘플 하향 조정
        return focal.mean()

# ──────────────────────────────────────────────
# Early Stopping
# ──────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 2, mode: str = 'max'):
        self.patience   = patience
        self.mode       = mode
        self.counter    = 0
        self.best_score = None

    def step(self, score: float) -> bool:
        """True 반환 시 학습 중단"""
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (score > self.best_score) if self.mode == 'max' \
                   else (score < self.best_score)

        if improved:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

# ──────────────────────────────────────────────
# 과적합 모니터
# ──────────────────────────────────────────────
class OverfittingMonitor:
    """
    에폭별 train/val loss 차이를 추적하여 과적합 상태를 판정.

    판정 기준
    ---------
    과소적합 : train loss > 0.4 이상 (모델이 학습 데이터에도 수렴 못함)
    정상      : val_loss - train_loss < 0.10
    과적합 주의 : gap 0.10~0.20, val loss 증가 추세
    과적합   : gap > 0.20 또는 val loss가 2에폭 연속 상승
    """
    GAP_WARNING  = 0.10
    GAP_OVERFIT  = 0.20
    UNDERFIT_THR = 0.40

    def __init__(self):
        self.history: list[dict] = []

    def update(self, epoch: int, train_loss: float, val_loss: float,
               val_f1: float) -> str:
        gap = val_loss - train_loss
        self.history.append({
            'epoch': epoch, 'train_loss': train_loss,
            'val_loss': val_loss, 'val_f1': val_f1, 'gap': gap,
        })

        # 상태 판정
        if train_loss > self.UNDERFIT_THR:
            status = "과소적합"
        elif gap > self.GAP_OVERFIT:
            status = "과적합"
        elif gap > self.GAP_WARNING and len(self.history) >= 2 \
                and val_loss > self.history[-2]['val_loss']:
            status = "과적합 주의"
        else:
            status = "정상"

        return status

    def summary(self) -> str:
        if not self.history:
            return ""
        lines = ["\n=== 과적합 진단 요약 ===",
                 f"{'에폭':>4}  {'Train':>7}  {'Val':>7}  {'Gap':>7}  상태"]
        for h in self.history:
            st = self.update.__func__(
                self, h['epoch'], h['train_loss'], h['val_loss'], h['val_f1']
            ) if False else ""
            lines.append(f"{h['epoch']:>4}  {h['train_loss']:>7.4f}  "
                         f"{h['val_loss']:>7.4f}  {h['gap']:>+7.4f}")
        return "\n".join(lines)

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class DisasterDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(
            texts.tolist(), truncation=True,
            padding='max_length', max_length=max_len, return_tensors='pt',
        )
        self.labels = torch.tensor(labels.tolist(), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()} | \
               {'labels': self.labels[idx]}

# ──────────────────────────────────────────────
# 평가
# ──────────────────────────────────────────────
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop('labels').to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            out    = model(**inputs)
            total_loss += criterion(out.logits, labels).item()
            all_preds.extend(out.logits.argmax(-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    n = len(loader)
    return (total_loss / n,
            accuracy_score(all_labels, all_preds),
            f1_score(all_labels, all_preds, average='macro',    zero_division=0),
            f1_score(all_labels, all_preds, average='weighted', zero_division=0),
            all_preds, all_labels)

# ──────────────────────────────────────────────
# 데이터 로드 및 샘플링
# ──────────────────────────────────────────────
print("\n데이터 로드 중...")
train_df = pd.read_csv('data_train.csv', encoding='utf-8-sig').fillna({'메시지내용': ''})
val_df   = pd.read_csv('data_val.csv',   encoding='utf-8-sig').fillna({'메시지내용': ''})
test_df  = pd.read_csv('data_test.csv',  encoding='utf-8-sig').fillna({'메시지내용': ''})
print(f"원본 Train {len(train_df):,} / Val {len(val_df):,} / Test {len(test_df):,}")

def stratified_sample(df, n_per_class, seed=42):
    parts = [df[df['label'] == lbl].sample(
                 min(len(df[df['label'] == lbl]), n_per_class), random_state=seed)
             for lbl in range(NUM_LABELS)]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

if MAX_SAMPLES is not None:
    train_df = stratified_sample(train_df, MAX_SAMPLES // NUM_LABELS)
    val_df   = stratified_sample(val_df,   max(200, MAX_SAMPLES // 10) // NUM_LABELS)
    print(f"샘플링 후 -> Train {len(train_df):,} / Val {len(val_df):,}")

print(f"Train 클래스 분포:\n{train_df['label_name'].value_counts().to_string()}\n")

# 클래스 가중치 (Focal Loss에도 동시 적용)
label_counts  = train_df['label'].value_counts().sort_index()
class_weights = torch.tensor(
    [len(train_df) / (NUM_LABELS * label_counts[i]) for i in range(NUM_LABELS)],
    dtype=torch.float,
).to(device)
print(f"클래스 가중치: {class_weights.cpu().numpy().round(3)}")

# ──────────────────────────────────────────────
# 토크나이저 & 데이터셋
# ──────────────────────────────────────────────
print(f"\n토크나이저 로드: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("토크나이징 중...")
train_ds = DisasterDataset(train_df['메시지내용'], train_df['label'], tokenizer, MAX_LENGTH)
val_ds   = DisasterDataset(val_df['메시지내용'],   val_df['label'],   tokenizer, MAX_LENGTH)
test_ds  = DisasterDataset(test_df['메시지내용'],  test_df['label'],  tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=(device.type=='cuda'))
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ──────────────────────────────────────────────
# 모델 / 옵티마이저 / 스케줄러
# ──────────────────────────────────────────────
print(f"\n모델 로드: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
model.to(device)

criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

total_steps  = len(train_loader) * MAX_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler    = get_cosine_schedule_with_warmup(   # Cosine decay (linear보다 안정)
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
)
print(f"총 스텝 {total_steps:,} / 워밍업 {warmup_steps:,}")

# ──────────────────────────────────────────────
# 학습 루프
# ──────────────────────────────────────────────
history = {k: [] for k in ['train_loss', 'val_loss', 'val_acc', 'val_macro_f1', 'gap']}
early_stop  = EarlyStopping(patience=ES_PATIENCE, mode='max')
of_monitor  = OverfittingMonitor()
best_val_f1 = 0.0
best_epoch  = 0
stop_reason = "최대 에폭 도달"

print(f"\n학습 시작 (max {MAX_EPOCHS} epochs, Early Stopping patience={ES_PATIENCE})")
print("="*65)
total_start = time.time()

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    ep_loss, ep_steps = 0.0, 0
    ep_start = time.time()

    for step, batch in enumerate(train_loader, 1):
        labels = batch.pop('labels').to(device)
        inputs = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        out  = model(**inputs)
        loss = criterion(out.logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        ep_loss  += loss.item()
        ep_steps += 1

        if step % 100 == 0:
            print(f"  E{epoch} step {step:>4}/{len(train_loader)} "
                  f"loss={ep_loss/ep_steps:.4f} ({time.time()-ep_start:.0f}s)")

    avg_train_loss = ep_loss / ep_steps
    val_loss, val_acc, val_f1, val_wf1, _, _ = evaluate(model, val_loader, criterion)

    gap    = val_loss - avg_train_loss
    status = of_monitor.update(epoch, avg_train_loss, val_loss, val_f1)

    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_macro_f1'].append(val_f1)
    history['gap'].append(gap)

    ep_time = time.time() - ep_start
    print(f"\nEpoch {epoch}/{MAX_EPOCHS}  ({ep_time:.0f}s)")
    print(f"  Train Loss : {avg_train_loss:.4f}")
    print(f"  Val   Loss : {val_loss:.4f}  Acc {val_acc:.4f}  Macro F1 {val_f1:.4f}")
    print(f"  Gap (val-train): {gap:+.4f}  -> [{status}]")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch  = epoch
        os.makedirs(SAVE_DIR, exist_ok=True)
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        print(f"  [Best] 저장 (Val Macro F1: {best_val_f1:.4f})")

    if early_stop.step(val_f1):
        stop_reason = f"Early Stopping (patience={ES_PATIENCE} 에폭 개선 없음)"
        print(f"\n  {stop_reason}")
        break
    print()

total_time = time.time() - total_start
actual_epochs = len(history['train_loss'])
print(f"\n학습 완료: {total_time/60:.1f}분 | Best epoch {best_epoch} | {stop_reason}")

# ──────────────────────────────────────────────
# Test 평가 (Best 모델)
# ──────────────────────────────────────────────
print("\nBest 모델로 Test 평가 중...")
model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR)
model.to(device)

test_loss, test_acc, test_f1, test_wf1, test_preds, test_labels = \
    evaluate(model, test_loader, criterion)

print("\n=== Test 결과 ===")
print(f"  Loss         : {test_loss:.4f}")
print(f"  Accuracy     : {test_acc:.4f}")
print(f"  Macro F1     : {test_f1:.4f}")
print(f"  Weighted F1  : {test_wf1:.4f}")
print("\n" + classification_report(test_labels, test_preds,
                                   target_names=LABEL_NAMES, zero_division=0))

# ──────────────────────────────────────────────
# 시각화 1: 과적합 진단 대시보드
# ──────────────────────────────────────────────
epochs_x = range(1, actual_epochs + 1)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('KoELECTRA-v3 학습 과정 및 과적합 진단', fontsize=14)

# (1) Loss 곡선
ax = axes[0, 0]
ax.plot(epochs_x, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
ax.plot(epochs_x, history['val_loss'],   'r-o', label='Val Loss',   linewidth=2)
ax.axvline(best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best (E{best_epoch})')
ax.fill_between(epochs_x, history['train_loss'], history['val_loss'],
                alpha=0.15, color='orange', label='Gap (과적합 지표)')
ax.set_title('Train / Val Loss')
ax.set_xlabel('Epoch')
ax.legend()
ax.grid(True, alpha=0.3)

# (2) Macro F1
ax = axes[0, 1]
ax.plot(epochs_x, history['val_macro_f1'], 'm-o', linewidth=2)
ax.axvline(best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best (E{best_epoch})')
ax.axhline(best_val_f1, color='gray', linestyle=':', alpha=0.5)
ax.set_title('Val Macro F1')
ax.set_xlabel('Epoch')
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True, alpha=0.3)

# (3) Gap 추이 (과적합 핵심 지표)
ax = axes[1, 0]
gap_colors = ['green' if g < OverfittingMonitor.GAP_WARNING
              else ('orange' if g < OverfittingMonitor.GAP_OVERFIT else 'red')
              for g in history['gap']]
bars = ax.bar(epochs_x, history['gap'], color=gap_colors, edgecolor='black', linewidth=0.5)
ax.axhline(OverfittingMonitor.GAP_WARNING, color='orange', linestyle='--',
           label=f'주의 기준 ({OverfittingMonitor.GAP_WARNING})', linewidth=1.5)
ax.axhline(OverfittingMonitor.GAP_OVERFIT, color='red',    linestyle='--',
           label=f'과적합 기준 ({OverfittingMonitor.GAP_OVERFIT})', linewidth=1.5)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title('Generalization Gap (Val - Train Loss)\n[녹색:정상 / 주황:주의 / 빨강:과적합]')
ax.set_xlabel('Epoch')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 에폭별 상태 레이블
for i, (ep, gap) in enumerate(zip(epochs_x, history['gap'])):
    st = of_monitor.update(ep, history['train_loss'][i],
                           history['val_loss'][i], history['val_macro_f1'][i])
    ax.text(ep, gap + 0.003, st, ha='center', fontsize=7, rotation=30)

# (4) Learning Rate 스케줄
ax = axes[1, 1]
lrs = [scheduler.get_last_lr()[0]] * actual_epochs  # 마지막 LR 기록용 (근사)
# Cosine 스케줄을 이론값으로 재현
import math
lr_curve = []
for s in range(len(train_loader) * actual_epochs):
    if s < warmup_steps:
        lr_curve.append(LR * s / max(1, warmup_steps))
    else:
        progress = (s - warmup_steps) / max(1, total_steps - warmup_steps)
        lr_curve.append(LR * 0.5 * (1 + math.cos(math.pi * progress)))

ax.plot(lr_curve, 'c-', linewidth=1)
ax.set_title('Learning Rate Schedule (Cosine Warmup)')
ax.set_xlabel('Step')
ax.set_ylabel('LR')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('koelectra_overfitting_diagnosis.png', dpi=150, bbox_inches='tight')
print("과적합 진단 저장: koelectra_overfitting_diagnosis.png")

# ──────────────────────────────────────────────
# 시각화 2: 혼동 행렬
# ──────────────────────────────────────────────
cm      = confusion_matrix(test_labels, test_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('KoELECTRA-v3 혼동 행렬 (Test)', fontsize=13)
for ax, data, fmt, title in zip(
    axes, [cm, cm_norm], ['d', '.2f'], ['절대값', '비율 (행 기준)']
):
    sns.heatmap(data, annot=True, fmt=fmt, cmap='Greens',
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_xlabel('예측')
    ax.set_ylabel('실제')
    ax.set_title(title)
plt.tight_layout()
plt.savefig('koelectra_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("혼동 행렬 저장: koelectra_confusion_matrix.png")

# ──────────────────────────────────────────────
# 전체 모델 비교
# ──────────────────────────────────────────────
try:
    results_df = pd.read_csv('results.csv', encoding='utf-8-sig')
except FileNotFoundError:
    results_df = pd.DataFrame()

new_row = pd.DataFrame([{
    'model':             f'KoELECTRA-v3 (E{best_epoch})',
    'val_acc':           max(history['val_acc']),
    'val_macro_f1':      best_val_f1,
    'test_acc':          test_acc,
    'test_macro_f1':     test_f1,
    'test_weighted_f1':  test_wf1,
    'train_time_sec':    round(total_time, 1),
}])
results_df = pd.concat([results_df, new_row], ignore_index=True)
results_df.to_csv('results.csv', index=False, encoding='utf-8-sig')

print("\n=== 전체 모델 비교 ===")
print(results_df[['model', 'test_acc', 'test_macro_f1', 'test_weighted_f1',
                   'train_time_sec']].to_string(index=False))

# 비교 시각화
if len(results_df) >= 2:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('모델 성능 비교 (Test Set)', fontsize=13)

    x  = np.arange(len(results_df))
    w  = 0.25
    c1, c2, c3 = '#3498db', '#e74c3c', '#2ecc71'

    for ax_idx, ax in enumerate(axes):
        if ax_idx == 0:
            b1 = ax.bar(x - w, results_df['test_acc'],         w, label='Accuracy',    color=c1)
            b2 = ax.bar(x,     results_df['test_macro_f1'],    w, label='Macro F1',    color=c2)
            b3 = ax.bar(x + w, results_df['test_weighted_f1'], w, label='Weighted F1', color=c3)
            ax.set_title('전체 성능 지표')
            ax.set_ylim(0.7, 1.05)
            for bar in [*b1, *b2, *b3]:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                        f'{bar.get_height():.3f}', ha='center', fontsize=7)
        else:
            # Macro F1만 강조 (불균형 대응 능력 핵심 지표)
            colors_bar = [c2] * len(results_df)
            bars = ax.barh(results_df['model'], results_df['test_macro_f1'],
                           color=colors_bar, edgecolor='black', linewidth=0.5)
            ax.set_xlim(0.7, 1.0)
            ax.set_title('Macro F1 비교 (불균형 대응 핵심 지표)')
            ax.set_xlabel('Macro F1')
            for bar, val in zip(bars, results_df['test_macro_f1']):
                ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                        f'{val:.4f}', va='center', fontsize=9)
        ax.legend(fontsize=8)
        if ax_idx == 0:
            ax.set_xticks(x)
            ax.set_xticklabels(results_df['model'], rotation=10, fontsize=8)

    plt.tight_layout()
    plt.savefig('all_models_comparison.png', dpi=150, bbox_inches='tight')
    print("전체 비교 저장: all_models_comparison.png")

# ──────────────────────────────────────────────
# 과적합 진단 텍스트 요약
# ──────────────────────────────────────────────
final_gap     = history['gap'][-1]
max_gap       = max(history['gap'])
max_gap_epoch = history['gap'].index(max_gap) + 1

if max_gap > OverfittingMonitor.GAP_OVERFIT:
    overall_status = f"과적합 감지 (최대 gap={max_gap:.4f}, Epoch {max_gap_epoch})"
elif max_gap > OverfittingMonitor.GAP_WARNING:
    overall_status = f"경미한 과적합 경향 (최대 gap={max_gap:.4f})"
else:
    overall_status = f"과적합 없음 (최대 gap={max_gap:.4f})"

summary = f"""
=== KoELECTRA-v3 학습 요약 ===
모델     : {MODEL_NAME}
손실함수 : Focal Loss (gamma={FOCAL_GAMMA})
학습방식 : Cosine Warmup LR / AdamW / Early Stopping (patience={ES_PATIENCE})
총 에폭  : {actual_epochs} / {MAX_EPOCHS} ({stop_reason})
Best 에폭: {best_epoch}
총 시간  : {total_time/60:.1f}분

=== 과적합 진단 ===
판정: {overall_status}

{'에폭':>4}  {'Train':>7}  {'Val':>7}  {'Gap':>7}  {'Macro F1':>9}
{'─'*46}"""
for i, ep in enumerate(range(1, actual_epochs + 1)):
    marker = " <-- Best" if ep == best_epoch else ""
    summary += (f"\n{ep:>4}  {history['train_loss'][i]:>7.4f}  "
                f"{history['val_loss'][i]:>7.4f}  {history['gap'][i]:>+7.4f}  "
                f"{history['val_macro_f1'][i]:>9.4f}{marker}")

summary += f"""

=== Test 최종 성능 ===
Accuracy    : {test_acc:.4f}
Macro F1    : {test_f1:.4f}
Weighted F1 : {test_wf1:.4f}
"""

print(summary)
with open('koelectra_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)
print("요약 저장: koelectra_summary.txt")

with open('koelectra_history.json', 'w') as f:
    json.dump(history, f, indent=2)

print("\n=== Step 4 완료 ===")
