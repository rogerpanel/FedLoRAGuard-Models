FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FEDLORAGUARD_CHECKPOINTS=/checkpoints

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY fedloraguard ./fedloraguard
COPY benchmarks   ./benchmarks
COPY baselines    ./baselines
COPY service      ./service
COPY configs      ./configs
COPY setup.py     ./
RUN pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
