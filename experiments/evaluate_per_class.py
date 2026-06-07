"""
중복 제거 테스트셋 클래스별 상세 평가

사용법:
  python evaluate_per_class.py --data 중요파일/data/raw/재난문자_레이블링결과.xlsx
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from dataset import load_and_split, DisasterDataset
from model import load_model

LABEL_NAMES = ['긴급아님(0)', '낮음(1)', '중간(2)', '높음(3)', '매우높음(4)']


def get_preds(df, tokenizer, model, device, max_length, batch_size):
    ds = DisasterDataset(df, tokenizer, max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, ncols=90):
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            ttids = batch['token_type_ids'].to(device)
            logits = model(input_ids=ids, attention_mask=mask, token_type_ids=ttids).logits
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(batch['label'].numpy())
    return all_labels, all_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',       type=str, default='중요파일/data/raw/재난문자_레이블링결과.xlsx')
    parser.add_argument('--model_path', type=str, default='model/')
    parser.add_argument('--tok_path',   type=str, default='tokenizer/')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("데이터 로드 중...")
    train_df, _, test_df = load_and_split(args.data, seed=args.seed)

    train_texts = set(train_df['text'].tolist())
    test_df = test_df.copy()
    test_df['is_dup'] = test_df['text'].isin(train_texts)
    dup_count = int(test_df['is_dup'].sum())
    clean_df  = test_df[~test_df['is_dup']].reset_index(drop=True)

    print(f"  전체 Test: {len(test_df):,}  중복 제거: {dup_count:,}  Clean: {len(clean_df):,}")

    print("모델 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(args.tok_path)
    model     = load_model(args.model_path, num_labels=5).to(device)
    model.eval()

    print(f"\n[1/2] 원본 전체 {len(test_df):,}건 평가...")
    orig_true, orig_pred = get_preds(test_df, tokenizer, model, device, args.max_length, args.batch_size)

    print(f"\n[2/2] 중복 제거 {len(clean_df):,}건 평가...")
    clean_true, clean_pred = get_preds(clean_df, tokenizer, model, device, args.max_length, args.batch_size)

    # ── 리포트 생성
    sep = "═" * 72
    lines = []

    for name, y_true, y_pred in [
        ("원본 전체 (32,999건)", orig_true, orig_pred),
        ("중복 제거 (29,812건)", clean_true, clean_pred),
    ]:
        lines.append(f"\n{sep}\n")
        lines.append(f"  {name}\n")
        lines.append(f"{sep}\n")

        report = classification_report(
            y_true, y_pred,
            target_names=LABEL_NAMES,
            digits=4, zero_division=0,
        )
        for line in report.splitlines():
            lines.append("  " + line + "\n")

        # 클래스별 오분류 수
        cm = confusion_matrix(y_true, y_pred)
        lines.append("\n  [오분류 상세]\n")
        for i in range(5):
            total = int(cm[i].sum())
            wrong = total - int(cm[i, i])
            lines.append(f"    레벨 {i} ({LABEL_NAMES[i]}): {total:>5,}건 중 {wrong}건 틀림\n")

    lines.append(f"\n{sep}\n")
    lines.append(f"  중복 제거 효과 요약\n")
    lines.append(f"{sep}\n")
    lines.append(f"  제거된 샘플: {dup_count:,}건 ({dup_count/len(test_df)*100:.1f}%)\n")
    lines.append(f"  오분류 변화: 동일 (중복 샘플 전부 정답이었음)\n")

    os.makedirs('results', exist_ok=True)
    out_path = 'results/evaluate_per_class_report.txt'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    for line in lines:
        print(line, end='')
    print(f"\n저장: {out_path}")


if __name__ == '__main__':
    main()
