import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

import os
import librosa
import whisper
from fastapi import FastAPI, UploadFile, File, BackgroundTasks

import moviepy.video.fx.all as fx
from moviepy.video.VideoClip import ColorClip, TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.fx.resize import resize
from thefuzz import process, fuzz
from google import genai
import json

app = FastAPI()
# This gets the directory where main.py actually lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(BASE_DIR, "fonts", "InterTight-VariableFont_wght.ttf")
# Configuration for paths
UPLOAD_DIR = "temp_files"
OUTPUT_DIR = "outputs"
# FONT_PATH = "C:/Windows/Fonts/arialbd.ttf" # Standard Windows Arial Bold path

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)



def get_pulse_scale(t, beat_times):
    """Calculates a zoom factor based on the music beats."""
    for beat in beat_times:
        if 0 <= (t - beat) < 0.15:
            return 1 + 0.2 * (1 - (t - beat) / 0.15)
    return 1.0

def text_clipper(txt, start, end):
    """Creates a TextClip with given text and timing."""
    duration = max(0.1, end - start)
    txt_clip = (TextClip(
                    txt=txt, 
                    fontsize=70, 
                    color='white', 
                    font=FONT_PATH,
                    method='caption',
                    size=(1000, 300),
                    align='center')
                .set_start(start)
                .set_duration(duration)
                .set_position('center'))
    return txt_clip


def audio_loader(audio_path: str):
    """Loads audio and returns AudioFileClip and beat times."""
    audio_clip = AudioFileClip(audio_path)
    y, sr = librosa.load(audio_path)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    return audio_clip, beat_times, tempo

def refine_lyrics_locally(whisper_result, original_lyrics_text):
    print("--- Refining Lyrics Locally (Fuzzy Matching) ---")
    
    # Split official lyrics into lines and remove empty ones
    official_lines = [line.strip() for line in original_lyrics_text.split('\n') if line.strip()]
    
    refined_segments = []
    
    for segment in whisper_result.get("segments", []):
        whisper_text = segment['text'].strip()
        
        # Find the best matching line from your official lyrics
        # limit=1 returns the single best match
        best_match, score = process.extractOne(whisper_text, official_lines, scorer=fuzz.token_sort_ratio)
        
        # If the match is decent (score > 50), use the official version
        # Otherwise, keep the whisper text to avoid weird jumps
        corrected_text = best_match if score > 50 else whisper_text
        
        refined_segments.append({
            "start": segment['start'],
            "end": segment['end'],
            "text": corrected_text
        })
        
    return refined_segments


def audio_transcriber(audio_path: str):
    """Transcribes audio using Whisper model."""
    print("--- Running Whisper Transcription ---")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, word_timestamps=True, fp16=False)
    return result

def process_video_task(audio_path: str, text_path: str, output_name: str):
    print(f"--- Starting Processing for {output_name} ---")
    
    try:
        # 1. LOAD AUDIO & ANALYZE BEATS
        print("--- Loading Audio and Analyzing Beats ---")
        audio_clip, beat_times, tempo = audio_loader(audio_path)
        
        bpm = tempo[0] if hasattr(tempo, "__len__") else tempo
        print(f"BPM Detected: {bpm}")

        # 2. WHISPER ALIGNMENT
        transcription = audio_transcriber(audio_path)
        
        # --- NEW REFINEMENT STEP ---
        # Read the official lyrics from the provided text file
        with open(text_path, "r", encoding="utf-8") as f:
            official_lyrics = f.read()
        
        # Apply fuzzy matching to align Whisper timestamps with official text
        refined_segments = refine_lyrics_locally(transcription, official_lyrics)
        # ---------------------------
        
        # 3. BACKGROUND LAYER
        bg = ColorClip(size=(1280, 720), color=(0, 255, 0)).set_duration(audio_clip.duration)
        clips = [bg]

        # 4. GENERATE SUBTITLE CLIPS (Looping through refined_segments now)
        for segment in refined_segments:
            txt = segment['text']
            
            txt_clip = text_clipper(
                txt=txt,
                start=segment['start'],
                end=segment['end']
            )

            # Apply Pulse effect
            txt_clip = txt_clip.fx(resize, lambda t: get_pulse_scale(t, beat_times))
            
            clips.append(txt_clip)

        # 5. FINAL COMPOSITION
        final_video = (CompositeVideoClip(clips, size=(1280, 720))
                       .set_audio(audio_clip)
                       .set_duration(audio_clip.duration))

        save_path = os.path.join(OUTPUT_DIR, output_name)
        
        # 6. RENDER
        print("--- Rendering Video File ---")
        final_video.write_videofile(
            save_path, 
            fps=24, 
            codec="libx264", 
            audio_codec="aac",
            threads=4
        )
        
        audio_clip.close()
        final_video.close()
        print(f"--- SUCCESS: {save_path} ---")

    except Exception as e:
        import traceback
        print(f"--- CRITICAL ERROR: {str(e)} ---")
        traceback.print_exc()
        
@app.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks, 
    audio: UploadFile = File(...), 
    text: UploadFile = File(...)
):
    audio_filename = audio.filename if audio.filename else "input_audio.mp3"
    text_filename = text.filename if text.filename else "input_text.txt"
    
    audio_path = os.path.join(UPLOAD_DIR, audio_filename)
    text_path = os.path.join(UPLOAD_DIR, text_filename)
    
    with open(audio_path, "wb") as f:
        f.write(await audio.read())
    with open(text_path, "wb") as f:
        f.write(await text.read())

    output_name = f"video_{audio_filename.split('.')[0]}.mp4"

    background_tasks.add_task(process_video_task, audio_path, text_path, output_name)

    return {
        "status": "Processing started",
        "message": "Check the server terminal for progress bars and output logs."
    }


@app.post("/transcribe-audio")
async def transcribe_audio_endpoint(audio: UploadFile = File(...), lyrics: UploadFile = File(None)):
    # 1. Save file
    audio_filename = audio.filename or "input_audio.mp3"
    audio_path = os.path.join(UPLOAD_DIR, audio_filename)
    lyrics_text = ""
    if lyrics:
        lyrics_content = await lyrics.read()
        lyrics_text = lyrics_content.decode('utf-8')


    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    whisper_result = audio_transcriber(audio_path)

    refined_result = refine_lyrics_locally(whisper_result, lyrics_text)

    return {
        "status": "success",
        "filename": audio_filename,
        "transcription": refined_result
    }