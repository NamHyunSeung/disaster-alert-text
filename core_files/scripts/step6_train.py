"""
Step 6: KoELECTRA-v3 최적화 학습

개선 사항
- WeightedRandomSampler : 배치마다 긴급 샘플 균등 노출 (복제 없음)
- Focal Loss + 클래스 가중치 : 불균형 이중 보정
- 임계값 최적화 : 학습 후 Val에서 긴급 최적 임계값 탐색
- batch_size 64 / max_length 96 : 속도 ~3.5× 향상
- 발신기관명 마스킹 : [기관명] → [기관] (어텐션 노이즈 제거)
- 범람 우려 레이블 보강 반영
- flush=True 전 프린트 : 실시간 출력 보장
"""

import os, sys, json, time, re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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

def log(msg):
    print(msg, flush=True)

# ─────────────────────────────────────────
# 발신기관명 마스킹
# ─────────────────────────────────────────
_ORG_PATTERN = re.compile(r'\[[^\]]{1,20}\]')

def normalize_text(text: str) -> str:
    """[창녕군청], [행정안전부] 등 발신기관명 → [기관] 통일"""
    return _ORG_PATTERN.sub('[기관]', text)

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
MODEL_NAME   = "../koelectra_v3_model"
MAX_LENGTH   = 96
BATCH_SIZE   = 64
MAX_EPOCHS   = 3
LR           = 3e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
FOCAL_GAMMA  = 2.0
ES_PATIENCE  = 2
NUM_LABELS   = 3
LABEL_NAMES  = ['긴급', '주의', '일반']
SAVE_DIR     = "../koelectra_v3_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"장치: {device}")
if device.type == "cuda":
    MAX_SAMPLES = None
    log(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    MAX_SAMPLES = 12_000
    log("[경고] CPU 모드: 12,000개 샘플")


# ─────────────────────────────────────────
# Focal Loss / EarlyStopping / 과적합 모니터
# ─────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
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
            self.best = score; self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class OverfittingMonitor:
    GAP_WARN    = 0.10
    GAP_OVERFIT = 0.20

    def __init__(self):
        self.history = []

    def update(self, epoch, tr, val, f1) -> str:
        gap = val - tr
        self.history.append(dict(epoch=epoch, tr=tr, val=val, gap=gap, f1=f1))
        if tr > 0.40:           return '과소적합'
        if gap > self.GAP_OVERFIT: return '과적합'
        if gap > self.GAP_WARN and len(self.history) >= 2 \
                and val > self.history[-2]['val']:
            return '과적합 주의'
        return '정상'


# ─────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────
class DisasterDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        log("  토크나이징 중...")
        self.enc = tokenizer(
            texts.tolist(), truncation=True,
            padding='max_length', max_length=MAX_LENGTH, return_tensors='pt',
        )
        self.labels = torch.tensor(labels.tolist(), dtype=torch.long)
        log(f"  완료: {len(self.labels):,}개")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.enc.items()} | \
               {'labels': self.labels[idx]}


# ─────────────────────────────────────────
# 평가
# ─────────────────────────────────────────
def evaluate(model, loader, criterion):
    model.eval()
    total, preds, labels, all_probs = 0.0, [], [], []
    with torch.no_grad():
        for batch in loader:
            y = batch.pop('labels').to(device)
            x = {k: v.to(device) for k, v in batch.items()}
            out = model(**x)
            total += criterion(out.logits, y).item()
            probs = F.softmax(out.logits, dim=-1).cpu().numpy()
            all_probs.extend(probs)
            preds.extend(probs.argmax(axis=1))
            labels.extend(y.cpu().numpy())
    n = len(loader)
    return (total / n,
            accuracy_score(labels, preds),
            f1_score(labels, preds, average='macro',    zero_division=0),
            f1_score(labels, preds, average='weighted', zero_division=0),
            preds, labels, np.array(all_probs))


# ─────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────
log("\n데이터 로드 중...")
train_df = pd.read_csv('../data/processed/data_train.csv', encoding='utf-8-sig').fillna({'메시지내용': ''})
val_df   = pd.read_csv('../data/processed/data_val.csv',   encoding='utf-8-sig').fillna({'메시지내용': ''})
test_df  = pd.read_csv('../data/processed/data_test.csv',  encoding='utf-8-sig').fillna({'메시지내용': ''})

# 발신기관명 마스킹 적용
for _df in [train_df, val_df, test_df]:
    _df['메시지내용'] = _df['메시지내용'].apply(normalize_text)
log("발신기관명 마스킹 적용: [기관명] -> [기관]")
log(f"Train {len(train_df):,} / Val {len(val_df):,} / Test {len(test_df):,}")


def stratified_sample(df, n_per_class, seed=42):
    parts = [df[df['label'] == l].sample(
        min(len(df[df['label'] == l]), n_per_class), random_state=seed)
        for l in range(NUM_LABELS)]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


if MAX_SAMPLES:
    train_df = stratified_sample(train_df, MAX_SAMPLES // NUM_LABELS)
    val_df   = stratified_sample(val_df,   max(300, MAX_SAMPLES // 10) // NUM_LABELS)
    log(f"샘플링 -> Train {len(train_df):,} / Val {len(val_df):,}")

log(f"\n[Train 분포]\n{train_df['label_name'].value_counts().to_string()}")

# 클래스 가중치
lc = train_df['label'].value_counts().sort_index()
class_weights = torch.tensor(
    [len(train_df) / (NUM_LABELS * lc[i]) for i in range(NUM_LABELS)],
    dtype=torch.float,
).to(device)
log(f"\n클래스 가중치: {dict(zip(LABEL_NAMES, class_weights.cpu().numpy().round(2)))}")

# WeightedRandomSampler
sample_weights = [class_weights[l].item() for l in train_df['label'].tolist()]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_df),
    replacement=True,
)
log("WeightedRandomSampler 적용: 배치마다 긴급 균등 노출")


# ─────────────────────────────────────────
# 토크나이저 & 데이터셋
# ─────────────────────────────────────────
log(f"\n토크나이저 로드: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

log("\n[Train]"); train_ds = DisasterDataset(train_df['메시지내용'], train_df['label'], tokenizer)
log("[Val]");   val_ds   = DisasterDataset(val_df['메시지내용'],   val_df['label'],   tokenizer)
log("[Test]");  test_ds  = DisasterDataset(test_df['메시지내용'],  test_df['label'],  tokenizer)

# sampler 사용 시 shuffle=False
train_loader = DataLoader(train_ds, BATCH_SIZE, sampler=sampler,
                          num_workers=0, pin_memory=(device.type == 'cuda'))
val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=0)


# ─────────────────────────────────────────
# 모델 / 옵티마이저 / 스케줄러
# ─────────────────────────────────────────
log(f"\n모델 로드: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
model.to(device)

criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

total_steps  = len(train_loader) * MAX_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler    = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
log(f"총 스텝 {total_steps:,} / 워밍업 {warmup_steps:,}")
log(f"스텝/에폭: {len(train_loader):,}  배치크기: {BATCH_SIZE}  max_length: {MAX_LENGTH}")


# ─────────────────────────────────────────
# 학습 루프
# ─────────────────────────────────────────
history    = {k: [] for k in ['train_loss', 'val_loss', 'val_acc',
                               'val_f1', 'val_긴급_f1', 'gap', 'step_losses']}
step_losses = []   # 스텝별 손실 (롤링 평균용)
es         = EarlyStopping(ES_PATIENCE)
ofm        = OverfittingMonitor()
best_f1    = 0.0
best_epoch = 0
stop_reason = "최대 에폭"

log(f"\n{'='*60}")
log(f"학습 시작 (max {MAX_EPOCHS} epochs, patience={ES_PATIENCE})")
log(f"{'='*60}")
t_total = time.time()

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
        step_losses.append(loss.item())

        if step % 50 == 0:
            elapsed = time.time() - ep_t
            remain  = elapsed / step * (len(train_loader) - step)
            log(f"  E{epoch} [{step:>4}/{len(train_loader)}] "
                f"loss={ep_loss/ep_n:.4f}  "
                f"경과 {elapsed:.0f}s  잔여 ~{remain:.0f}s")

    tr_loss = ep_loss / ep_n
    vl, va, vf, vwf, _, vlabels, vprobs = evaluate(model, val_loader, criterion)
    vf_emerg = f1_score(vlabels, vprobs.argmax(axis=1),
                        labels=[0], average='macro', zero_division=0)
    gap    = vl - tr_loss
    status = ofm.update(epoch, tr_loss, vl, vf)

    history['train_loss'].append(tr_loss)
    history['val_loss'].append(vl)
    history['val_acc'].append(va)
    history['val_f1'].append(vf)
    history['val_긴급_f1'].append(vf_emerg)
    history['gap'].append(gap)
    history['step_losses'] = step_losses.copy()

    ep_time = time.time() - ep_t
    log(f"\n{'─'*55}")
    log(f"Epoch {epoch}/{MAX_EPOCHS}  ({ep_time:.0f}s / {ep_time/60:.1f}min)")
    log(f"  Train Loss : {tr_loss:.4f}")
    log(f"  Val   Loss : {vl:.4f}  Acc {va:.4f}  Macro F1 {vf:.4f}")
    log(f"  긴급 F1    : {vf_emerg:.4f}")
    log(f"  Gap        : {gap:+.4f}  [{status}]")

    if vf > best_f1:
        best_f1, best_epoch = vf, epoch
        os.makedirs(SAVE_DIR, exist_ok=True)
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        log(f"  [Best 저장] Val Macro F1: {best_f1:.4f}")

    if es.step(vf):
        stop_reason = f"Early Stopping (patience={ES_PATIENCE})"
        log(f"\n  {stop_reason}")
        break
    log("")

total_time = time.time() - t_total
actual_ep  = len(history['train_loss'])
log(f"\n학습 완료: {total_time/60:.1f}분 | Best Epoch {best_epoch} | {stop_reason}")


# ─────────────────────────────────────────
# Test 평가 (Best 모델)
# ─────────────────────────────────────────
log("\nBest 모델 로드 후 Test 평가...")
model = AutoModelForSequenceClassification.from_pretrained(SAVE_DIR)
model.to(device)

_, ta, tf, twf, t_preds, t_labels, t_probs = evaluate(model, test_loader, criterion)
log(f"\n=== Test 결과 (기본 임계값 0.5) ===")
log(f"  Accuracy  {ta:.4f}  Macro F1  {tf:.4f}  Weighted F1  {twf:.4f}")
log("\n" + classification_report(t_labels, t_preds, target_names=LABEL_NAMES, zero_division=0))


# ─────────────────────────────────────────
# 임계값 최적화 (Val 기준 긴급 F1 최대화)
# ─────────────────────────────────────────
log("\n임계값 최적화 중 (Val 세트)...")
_, _, _, _, _, val_labels_all, val_probs_all = evaluate(model, val_loader, criterion)
val_labels_all = np.array(val_labels_all)

thresh_results = []
for t in np.arange(0.10, 0.90, 0.02):
    preds_t = np.where(val_probs_all[:, 0] >= t, 0,
                       val_probs_all[:, 1:].argmax(axis=1) + 1)
    f1_e = f1_score(val_labels_all, preds_t, labels=[0], average='macro', zero_division=0)
    f1_m = f1_score(val_labels_all, preds_t, average='macro', zero_division=0)
    thresh_results.append((t, f1_e, f1_m))

best_thresh = max(thresh_results, key=lambda x: x[1] + x[2])[0]
log(f"최적 임계값: {best_thresh:.2f}  "
    f"(긴급 F1: {max(thresh_results, key=lambda x: x[1])[1]:.4f})")

# 최적 임계값으로 Test 평가
t_preds_opt = np.where(t_probs[:, 0] >= best_thresh, 0,
                        t_probs[:, 1:].argmax(axis=1) + 1)
ta_opt  = accuracy_score(t_labels, t_preds_opt)
tf_opt  = f1_score(t_labels, t_preds_opt, average='macro',    zero_division=0)
twf_opt = f1_score(t_labels, t_preds_opt, average='weighted', zero_division=0)

log(f"\n=== Test 결과 (최적 임계값 {best_thresh:.2f}) ===")
log(f"  Accuracy  {ta_opt:.4f}  Macro F1  {tf_opt:.4f}  Weighted F1  {twf_opt:.4f}")
log("\n" + classification_report(t_labels, t_preds_opt,
                                 target_names=LABEL_NAMES, zero_division=0))


# ─────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────
ep_x = range(1, actual_ep + 1)
colors = ['#e74c3c', '#f39c12', '#3498db']

# 1. 학습 대시보드 (4개 패널)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('KoELECTRA-v3 v2 학습 과정', fontsize=14)

# Loss 곡선
ax = axes[0, 0]
ax.plot(ep_x, history['train_loss'], 'b-o', label='Train')
ax.plot(ep_x, history['val_loss'],   'r-o', label='Val')
ax.fill_between(ep_x, history['train_loss'], history['val_loss'],
                alpha=0.15, color='orange', label='Gap')
ax.axvline(best_epoch, color='green', linestyle='--', label=f'Best E{best_epoch}')
ax.set_title('Train / Val Loss')
ax.legend(); ax.grid(alpha=0.3)

# Val 지표
ax = axes[0, 1]
ax.plot(ep_x, history['val_f1'],       'm-o', label='Macro F1')
ax.plot(ep_x, history['val_긴급_f1'], 'r-s', label='긴급 F1', linewidth=2)
ax.plot(ep_x, history['val_acc'],      'g-^', label='Accuracy')
ax.axvline(best_epoch, color='gray', linestyle='--')
ax.set_ylim(0, 1); ax.set_title('Val 성능 지표')
ax.legend(); ax.grid(alpha=0.3)

# Gap (과적합 진단)
ax = axes[1, 0]
gc = ['green' if g < ofm.GAP_WARN
      else ('orange' if g < ofm.GAP_OVERFIT else 'red')
      for g in history['gap']]
ax.bar(ep_x, history['gap'], color=gc)
ax.axhline(ofm.GAP_WARN,    color='orange', linestyle='--', label=f'주의({ofm.GAP_WARN})')
ax.axhline(ofm.GAP_OVERFIT, color='red',    linestyle='--', label=f'과적합({ofm.GAP_OVERFIT})')
for i, (ep, h) in enumerate(zip(ep_x, ofm.history)):
    ax.text(ep, history['gap'][i] + 0.003, h['epoch'] and
            ofm.update(h['epoch'], h['tr'], h['val'], h['f1']),
            ha='center', fontsize=7, rotation=20)
ax.set_title('Generalization Gap')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# 임계값 탐색
ax = axes[1, 1]
ts  = [r[0] for r in thresh_results]
f1e = [r[1] for r in thresh_results]
f1m = [r[2] for r in thresh_results]
ax.plot(ts, f1e, 'r-o', label='긴급 F1', markersize=4)
ax.plot(ts, f1m, 'm-s', label='Macro F1', markersize=4)
ax.axvline(best_thresh, color='green', linestyle='--', label=f'최적({best_thresh:.2f})')
ax.set_title('긴급 임계값 최적화 (Val)')
ax.set_xlabel('임계값')
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../../outputs/koelectra_v3_v2_dashboard.png', dpi=150, bbox_inches='tight')
log("대시보드 저장: ../../outputs/koelectra_v3_v2_dashboard.png")

# 2. 혼동 행렬 (기본 / 최적 임계값)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('혼동 행렬 비교', fontsize=13)
for row, (preds, title) in enumerate([(t_preds, f'기본 임계값 0.5'),
                                       (t_preds_opt, f'최적 임계값 {best_thresh:.2f}')]):
    cm      = confusion_matrix(t_labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    for col, (data, fmt, sub) in enumerate([(cm, 'd', '절대값'),
                                             (cm_norm, '.2f', '비율')]):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
                    ax=axes[row, col])
        axes[row, col].set_title(f'{title} - {sub}')
        axes[row, col].set_xlabel('예측')
        axes[row, col].set_ylabel('실제')
plt.tight_layout()
plt.savefig('../../outputs/koelectra_v3_v2_confusion.png', dpi=150, bbox_inches='tight')
log("혼동 행렬 저장: ../../outputs/koelectra_v3_v2_confusion.png")

# 3. 전체 모델 비교
try:
    results_df = pd.read_csv('results.csv', encoding='utf-8-sig')
except FileNotFoundError:
    results_df = pd.DataFrame()

new_row = pd.DataFrame([{
    'model':             f'KoELECTRA-v3 (E{best_epoch}, thresh={best_thresh:.2f}, masked)',
    'val_acc':           max(history['val_acc']),
    'val_macro_f1':      best_f1,
    'test_acc':          ta_opt,
    'test_macro_f1':     tf_opt,
    'test_weighted_f1':  twf_opt,
    'train_time_sec':    round(total_time, 1),
}])
results_df = pd.concat([results_df, new_row], ignore_index=True)
results_df.to_csv('results.csv', index=False, encoding='utf-8-sig')

log("\n=== 전체 모델 비교 ===")
log(results_df[['model', 'test_acc', 'test_macro_f1', 'test_weighted_f1']].to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(results_df['model'], results_df['test_macro_f1'],
               color='#e74c3c', edgecolor='black', linewidth=0.5)
ax.set_xlim(0.7, 1.0)
ax.set_title('Test Macro F1 비교')
ax.set_xlabel('Macro F1')
for bar, v in zip(bars, results_df['test_macro_f1']):
    ax.text(v + 0.002, bar.get_y() + bar.get_height()/2,
            f'{v:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('../../outputs/koelectra_v3_v2_all_models_comparison.png', dpi=150, bbox_inches='tight')
log("비교 저장: ../../outputs/koelectra_v3_v2_all_models_comparison.png")

# ─────────────────────────────────────────
# 요약 저장
# ─────────────────────────────────────────
max_gap   = max(history['gap'])
of_status = ('과적합' if max_gap > ofm.GAP_OVERFIT
             else '과적합 주의' if max_gap > ofm.GAP_WARN else '정상')

summary = f"""
=== KoELECTRA-v3 v2 학습 요약 ===
모델        : {MODEL_NAME}
설정        : batch={BATCH_SIZE} / max_len={MAX_LENGTH} / lr={LR}
개선 사항   : WeightedRandomSampler + Focal Loss(γ={FOCAL_GAMMA}) + 임계값 최적화
총 에폭     : {actual_ep}/{MAX_EPOCHS}  ({stop_reason})
Best 에폭   : {best_epoch}  (Val Macro F1: {best_f1:.4f})
학습 시간   : {total_time/60:.1f}분

=== 과적합 진단: {of_status} (최대 gap={max_gap:.4f}) ===
{'에폭':>4}  {'Train':>7}  {'Val':>7}  {'Gap':>7}  {'Macro F1':>9}  {'긴급 F1':>8}
{'─'*56}"""
for h in ofm.history:
    m = " <- Best" if h['epoch'] == best_epoch else ""
    ef = history['val_긴급_f1'][h['epoch']-1]
    summary += f"\n{h['epoch']:>4}  {h['tr']:>7.4f}  {h['val']:>7.4f}  " \
               f"{h['gap']:>+7.4f}  {h['f1']:>9.4f}  {ef:>8.4f}{m}"

summary += f"""

=== Test 최종 성능 ===
기본 임계값 0.5 : Accuracy {ta:.4f}  Macro F1 {tf:.4f}
최적 임계값 {best_thresh:.2f}: Accuracy {ta_opt:.4f}  Macro F1 {tf_opt:.4f}
최적 임계값     : {best_thresh:.2f}
"""

with open('../../outputs/koelectra_v3_v2_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)
log(summary)

with open('../../outputs/koelectra_v3_v2_history.json', 'w') as f:
    h = {k: v for k, v in history.items() if k != 'step_losses'}
    json.dump(h, f, indent=2)

log("\n=== Step 6 완료 ===")
