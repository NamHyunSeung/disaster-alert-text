"""
모델을 Hugging Face Hub에 업로드

사용법:
  1. pip install huggingface_hub
  2. huggingface-cli login   (HF 계정 토큰 입력)
  3. python push_to_hub.py
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR    = "koelectra_v3_model"
HUB_MODEL_ID = input("HF Hub 모델 ID 입력 (예: your-username/koelectra-disaster-v3): ").strip()

print("토크나이저 업로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
tokenizer.push_to_hub(HUB_MODEL_ID)

print("모델 업로드 중...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.push_to_hub(HUB_MODEL_ID)

print(f"\n완료: https://huggingface.co/{HUB_MODEL_ID}")
print(f"hf_space/app.py 에서 HUB_MODEL_ID = \"{HUB_MODEL_ID}\" 로 수정하세요.")
