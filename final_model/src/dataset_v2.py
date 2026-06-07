"""
Level 2 중복 제거 데이터셋 + 키워드 마스킹 Augmentation
"""

import re
import random
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from dataset import preprocess_text

# 긴 키워드 먼저 (부분 치환 방지)
_MASK_KEYWORDS = sorted([
    '즉시 대피', '대피명령', '대피 명령', '긴급대피', '긴급 대피', '신속히 대피',
    '지진 발생', '쓰나미', '민방공', '테러',
    '경보', '주의보', '특보', '예비특보',
    '대피', '발생', '화재', '산불', '홍수', '태풍', '침수', '범람',
    '해제', '종료', '완료',
], key=len, reverse=True)

_SPACES = re.compile(r'\s+')


def _mask_text(text: str) -> str:
    for kw in _MASK_KEYWORDS:
        text = text.replace(kw, ' ')
    return _SPACES.sub(' ', text).strip()


def _mask_text_partial(text: str, mask_ratio: float = 0.6) -> str:
    present = [kw for kw in _MASK_KEYWORDS if kw in text]
    if not present:
        return text
    n_mask = max(1, round(len(present) * mask_ratio))
    to_mask = sorted(random.sample(present, min(n_mask, len(present))), key=len, reverse=True)
    for kw in to_mask:
        text = text.replace(kw, ' ')
    return _SPACES.sub(' ', text).strip()


def load_and_split_v2(
    data_path: str = '중요파일/data/raw/재난문자_레이블링결과_dedup_v2.xlsx',
    seed: int = 42,
):
    df = pd.read_excel(data_path)
    df = df[df['label'] != -1].copy()
    df['label'] = df['label'].astype(int)
    df['text'] = df['메시지내용'].fillna('').apply(preprocess_text)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df['label'], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df['label'], random_state=seed
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


class DisasterDatasetAug(Dataset):
    """augment=True이면 학습 시 30% 확률로 키워드 마스킹 적용."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128,
                 augment: bool = False, mask_prob: float = 0.3,
                 return_text: bool = False):
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.mask_prob = mask_prob
        self.return_text = return_text

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.augment and random.random() < self.mask_prob:
            text = _mask_text(text)

        enc = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        result = {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'token_type_ids': enc.get(
                'token_type_ids',
                torch.zeros(self.max_length, dtype=torch.long)
            ).squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if self.return_text:
            result['text'] = self.texts[idx]  # 원본 텍스트 (augment 전)
        return result


def v9_collate_fn(batch):
    """return_text=True 배치 처리용 collate. text는 list로 분리."""
    from torch.utils.data.dataloader import default_collate
    texts = [item.pop('text') for item in batch]
    collated = default_collate(batch)
    collated['text'] = texts
    return collated
