import re
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

_ORG_PATTERN = re.compile(r'\[[^\]]{1,20}\]')


def preprocess_text(text: str) -> str:
    text = str(text)
    text = _ORG_PATTERN.sub(' ', text)
    text = text.replace('\n', ' ')
    return text.strip()


def load_and_split(data_path: str, seed: int = 42):
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


class DisasterDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'token_type_ids': enc.get(
                'token_type_ids',
                torch.zeros(self.max_length, dtype=torch.long)
            ).squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
        }
