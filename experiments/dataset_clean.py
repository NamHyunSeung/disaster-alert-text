"""
중복 제거 데이터셋 로드 및 분할

사용법:
  from dataset_clean import load_and_split_clean, DisasterDataset
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import preprocess_text, DisasterDataset  # noqa: F401


def load_and_split_clean(
    data_path: str = '중요파일/data/raw/재난문자_레이블링결과_dedup.xlsx',
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
