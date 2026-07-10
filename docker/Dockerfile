FROM python:3.10.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
EXPOSE 8501
CMD ["streamlit","run", "src/app.py"]