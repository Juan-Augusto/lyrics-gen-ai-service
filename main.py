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
        # If current time 't' is within 150ms after a beat hit
        if 0 <= (t - beat) < 0.15:
            # Linear scaling from 1.2 down to 1.0 over 150ms
            return 1 + 0.2 * (1 - (t - beat) / 0.15)
    return 1.0

def process_video_task(audio_path: str, text_path: str, output_name: str):
    print(f"--- Starting Processing for {output_name} ---")
    
    try:
        # 1. LOAD AUDIO & ANALYZE BEATS
        audio_clip = AudioFileClip(audio_path)
        y, sr = librosa.load(audio_path)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        
        bpm = tempo[0] if hasattr(tempo, "__len__") else tempo
        print(f"BPM Detected: {bpm}")

        # 2. WHISPER ALIGNMENT
        print("--- Running Whisper Transcription ---")
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, word_timestamps=True, fp16=False)
        segments = result.get('segments', [])
        
        # 3. BACKGROUND LAYER
        # AUDIT: Changed .with_duration to .set_duration
        bg = ColorClip(size=(1280, 720), color=(0, 255, 0)).set_duration(audio_clip.duration)
        clips = [bg]

        # 4. GENERATE SUBTITLE CLIPS
        for segment in segments:
            txt = segment['text'].strip()
            
            # AUDIT: Using 'txt', 'fontsize', and 'set_' methods per your diagnostic logs
            txt_clip = (TextClip(
                            txt=txt, 
                            fontsize=70, 
                            color='white', 
                            font=FONT_PATH,
                            method='caption',
                            size=(1000, 300),
                            align='center')
                        .set_start(segment['start'])
                        .set_duration(max(0.1, segment['end'] - segment['start']))
                        .set_position('center'))

            # AUDIT: Changed .with_effects to .set_effects (standard for dev2)
            # Also using 'fx.Resize' module import
            txt_clip = txt_clip.fx(resize, lambda t: get_pulse_scale(t, beat_times))
            
            clips.append(txt_clip)

        # 5. FINAL COMPOSITION
        # AUDIT: Changed .with_audio and .with_duration to .set_audio and .set_duration
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
        traceback.print_exc() # This will show us EXACTLY which line failed

        
@app.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks, 
    audio: UploadFile = File(...), 
    text: UploadFile = File(...)
):
    # Save the files
    audio_filename = audio.filename if audio.filename else "input_audio.mp3"
    text_filename = text.filename if text.filename else "input_text.txt"
    
    audio_path = os.path.join(UPLOAD_DIR, audio_filename)
    text_path = os.path.join(UPLOAD_DIR, text_filename)
    
    with open(audio_path, "wb") as f:
        f.write(await audio.read())
    with open(text_path, "wb") as f:
        f.write(await text.read())

    # Generate unique output name
    output_name = f"video_{audio_filename.split('.')[0]}.mp4"

    # Queue the background task
    background_tasks.add_task(process_video_task, audio_path, text_path, output_name)

    return {
        "status": "Processing started",
        "message": "Check the server terminal for progress bars and output logs."
    }