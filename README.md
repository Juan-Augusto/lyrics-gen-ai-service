## Important commands

Create a new venv: python -m venv venv
Activate: venv\Scripts\activate

Install dependencies: pip install fastapi uvicorn librosa moviepy openai-whisper python-multipart
Run: uvicorn main:app --reload

## Running container

Build: docker build -t lyrics-ai-be .
Run: docker run -d -p 8000:8000 lyrics-ai-be

docker-compose build
docker-compose up
