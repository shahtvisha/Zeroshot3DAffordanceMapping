FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    fastapi uvicorn[standard] python-multipart

COPY . .

RUN mkdir -p web/demo_scenes results/tables results/figures

EXPOSE 8000

CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "7860"]
