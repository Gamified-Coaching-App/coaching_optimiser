FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev \
    python3-dev \
    pkg-config \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY src/inference_pipeline/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY src/inference_pipeline /app
COPY src/training_data/training_data.py /app/training_data/training_data.py

EXPOSE 80

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"] 