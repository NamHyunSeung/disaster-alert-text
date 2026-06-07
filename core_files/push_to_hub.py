"""
모델을 Hugging Face Hub에 업로드 (KoELECTRA v3 5-class)

사용법:
  1. pip install huggingface_hub
  2. huggingface-cli login   (HF 계정 토큰 입력)
  3. python push_to_hub.py
  4. python push_to_hub.py --id nhs0327/koelectra-disaster-v9n  (ID 직접 지정)
"""

import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument("--id", default="nhs0327/koelectra-disaster-v9n",
                    help="HF Hub 모델 ID")
args = parser.parse_args()

MODEL_DIR    = "model_v9n"
TOK_DIR      = "tokenizer_v9n"
HUB_MODEL_ID = args.id

print("토크나이저 업로드 중...")
tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)
tokenizer.push_to_hub(HUB_MODEL_ID)

print("모델 업로드 중...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.push_to_hub(HUB_MODEL_ID)

print(f"\n완료: https://huggingface.co/{HUB_MODEL_ID}")
print(f"hf_space/app.py 에서 HUB_MODEL_ID = \"{HUB_MODEL_ID}\" 로 수정하세요.")
