FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi>=0.111.0 \
    uvicorn>=0.30.0 \
    transformers>=4.40.0 \
    torch>=2.1.0 \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY app.py .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
