# Use a lightweight Python base
FROM python:3.10-slim

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
  ffmpeg \
  imagemagick \
  && rm -rf /var/lib/apt/lists/*

# Fix ImageMagick security policy - Using wildcard to find the correct version folder
RUN sed -i 's/domain="path" rights="none" pattern="@\*"/domain="path" rights="read|write" pattern="@*"/g' /etc/ImageMagick-*/policy.xml
WORKDIR /app

# 1. Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt  
# Usually takes >1000s
#RUN pip install -r requirements.txt

# 2. Pre-download Whisper
RUN python -c "import whisper; whisper.load_model('base')"

# 3. Copy the rest of the project
COPY . .

# 4. Ensure the fonts directory is available
# We don't need to copy to /usr/share/ anymore because 
# your main.py now looks in the local /app/fonts folder.
RUN mkdir -p /app/fonts

# Expose and Run
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]