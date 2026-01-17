FROM python:3.12-slim

WORKDIR /app

# Install system dependencies from packages.txt + extras
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    libtesseract-dev \
    tesseract-ocr-eng \
    tesseract-ocr-spa \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY constraints.txt .

# Install Python packages with constraints
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Copy all application files
COPY . .

# Expose Hugging Face port
EXPOSE 7860

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]