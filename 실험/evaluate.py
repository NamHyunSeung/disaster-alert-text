"""
저장된 모델로 평가만 실행

사용법:
  python evaluate.py --model_path model/ --data 재난문자_레이블링결과.xlsx
"""

import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from dataset import load_and_split, DisasterDataset
from model import load_model
from utils import (
    print_evaluation, save_evaluation_report,
    save_confusion_matrix,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='model/',     help='저장된 모델 경로')
    parser.add_argument('--tok_path',   type=str, default='tokenizer/', help='저장된 토크나이저 경로')
    parser.add_argument('--data',       type=str, required=True,        help='엑셀 파일 경로')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("데이터 로드 중...")
    _, _, test_df = load_and_split(args.data, seed=args.seed)
    print(f"Test: {len(test_df):,}건")

    print("토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(args.tok_path)

    test_ds     = DisasterDataset(test_df, tokenizer, args.max_length)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("모델 로드 중...")
    model = load_model(args.model_path, num_labels=5).to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Test]", ncols=100):
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            ttids = batch['token_type_ids'].to(device)
            lbls  = batch['label']

            logits = model(input_ids=ids, attention_mask=mask, token_type_ids=ttids).logits
            preds  = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbls.numpy())

    print_evaluation(all_labels, all_preds)

    import os
    os.makedirs('results', exist_ok=True)
    save_evaluation_report(all_labels, all_preds, 'results/evaluation_report.txt')
    save_confusion_matrix(all_labels, all_preds, 'results/confusion_matrix.png')
    print("\n결과 저장: results/evaluation_report.txt / confusion_matrix.png")


if __name__ == '__main__':
    main()
