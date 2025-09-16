import os
import subprocess
import stable_whisper
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
import numpy as np
import glob
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import multiprocessing
import tempfile
import uuid
import shutil
import psutil

if not hasattr(np, "NaN"):
    np.NaN = np.nan


def extract_audio(video_path):
    # Create a unique temporary directory for this job
    temp_dir = os.path.join(tempfile.gettempdir(), f"subgen_{uuid.uuid4().hex}")
    os.makedirs(temp_dir, exist_ok=True)

    # Create unique audio file path
    audio_path = os.path.join(temp_dir, "audio.wav")

    """Extract audio from video using ffmpeg."""
    command = [
        "ffmpeg",
        "-y",  # overwrite if exists
        "-i", video_path,
        "-vn",  # no video
        "-ac", "1",  # mono
        "-ar", "16000",  # 16k sample rate
        "-acodec", "pcm_s16le",  # ✅ ensure raw 16-bit PCM
        "-af", "loudnorm,highpass=f=200,lowpass=f=8000",
        audio_path,
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path, temp_dir

def write_subtitles(transcription, sub_format, filename):
    """Write subtitles to SRT or VTT format."""
    lines = []

    if sub_format == "srt":
        for i, seg in enumerate(transcription, start=1):
            start = seg["start"]
            end = seg["end"]
            text = seg["text"]

            start_time = format_timestamp(start, srt=True)
            end_time = format_timestamp(end, srt=True)

            lines.append(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")

    elif sub_format == "vtt":
        lines.append("WEBVTT\n\n")
        for seg in transcription:
            start = seg["start"]
            end = seg["end"]
            text = seg["text"]

            start_time = format_timestamp(start, srt=False)
            end_time = format_timestamp(end, srt=False)

            lines.append(f"{start_time} --> {end_time}\n{text}\n\n")

    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(lines)


def format_timestamp(seconds, srt=True):
    ms = int((seconds % 1) * 1000)
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)

    if srt:
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    else:
        return f"{h:02}:{m:02}:{s:02}.{ms:03}"


import subprocess
import os

def split_audio(audio_path, chunk_length=600, overlap=1.0):  # default 600s = 10 min
    """
    Splits audio into fixed-length chunks (default 10 min).
    Always re-encodes into 16kHz mono PCM WAV to avoid muxer errors.
    """
    # Get base directory of audio file
    temp_dir = os.path.dirname(audio_path)

    # Get audio duration using ffprobe
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", audio_path
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    try:
        duration = float(result.stdout.strip())
    except Exception as e:
        raise RuntimeError(f"Failed to get duration for {audio_path}: {result.stdout}") from e

    chunks = []
    # Calculate number of chunks with overlap
    step = chunk_length - overlap
    num_chunks = math.ceil(duration / step)

    for i in range(num_chunks):
        start_time = i * step
        end_time = min(start_time + chunk_length, duration)
        actual_length = end_time - start_time

        chunk_file = f"{os.path.splitext(audio_path)[0]}_part{i}.wav"
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time), "-t", str(actual_length),
            "-i", audio_path,
            "-ar", "16000", "-ac", "1",  # re-encode for consistency
            "-c:a", "pcm_s16le",  # 16-bit PCM format
            chunk_file
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Skip if the chunk is too small or empty
        if os.path.getsize(chunk_file) < 1000:
            print(f"Skipping empty/small chunk: {chunk_file}")
            continue

        chunks.append((chunk_file, start_time))

    return chunks

def batch_translate_m2m100(texts, src_lang, tgt_lang, batch_size=8):
    """
    Translate a list of subtitle texts in smaller batches using M2M100.
    Prevents memory issues with long transcripts.
    """

    model_name = "C:\\Users\\42810\\Desktop\\m2m100_1.2B"
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    tokenizer.src_lang = src_lang

    results = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    print(f"[INFO] Translating {len(texts)} segments in {total_batches} batches...")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"  🔄 Translating batch {i//batch_size+1}/{total_batches} "
              f"({len(batch)} segments) ...", flush=True)
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            generated_tokens = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
                max_length=512
            )
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        results.extend(decoded)

    return results

def transcribe_single_chunk(args):
    try:
        chunk_file, offset, model_size, task, language = args
        model = stable_whisper.load_model(model_size, device="cpu")  # force CPU
        result = model.transcribe(chunk_file, task=task, language=language)
        result_dict = result.to_dict()

        # ✅ Safety check
        if not isinstance(result_dict, dict) or "segments" not in result_dict:
            print(f"⚠️ No valid segments for chunk {chunk_file}, skipping.")
            return []
        
        segs = result_dict["segments"]
        # adjust timestamps
        for seg in segs:
            seg["start"] += offset
            seg["end"] += offset
        return segs
    except Exception as e:
        print(f"⚠️ Error in chunk {chunk_file}: {e}")
        return []

def merge_overlapping_segments(segments, overlap=1.0):
    """
    Merge transcription segments from overlapping chunks.
    
    Args:
        segments (list): List of segment dicts with "start", "end", "text".
        overlap (float): Overlap used during audio splitting (seconds).
    
    Returns:
        list: Merged list of segments without duplicates.
    """
    if not segments:
        return []

    merged_segments = []
    prev_end = -float("inf")

    for seg in segments:
        # If this segment overlaps with previous due to chunk overlap
        if seg["start"] < prev_end:
            # Avoid duplicate text
            # Keep the one with longer duration (better coverage)
            if merged_segments and (seg["end"] - seg["start"]) > (merged_segments[-1]["end"] - merged_segments[-1]["start"]):
                merged_segments[-1] = seg  # replace previous with current
        else:
            merged_segments.append(seg)

        prev_end = seg["end"]
    for seg in merged_segments:
        seg["text"] = clean_text(seg["text"])
    return merged_segments

import re
def clean_text(text):
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces/newlines
    return text

def transcribe_chunks_parallel(chunks, model_size="medium", task="transcribe", language=None):
    # max_workers = max(1, multiprocessing.cpu_count() // 2)
    total_cpus = multiprocessing.cpu_count()
    cpu_load = psutil.cpu_percent(interval=0.1)
    if cpu_load > 70:  # system already busy
        max_workers = max(1, total_cpus // 4)  # be conservative
    else:
        max_workers = max(1, total_cpus // 2)

    all_segments = []

    print(f"Total chunks: {len(chunks)} (using {max_workers} workers on CPU)")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                transcribe_single_chunk,
                (chunk_file, offset, model_size, task, language)
            )
            for chunk_file, offset in chunks
        ]

        for i, f in enumerate(as_completed(futures), start=1):
            try:
                segs = f.result()
                all_segments.extend(segs)
                print(f"Finished chunk {i}/{len(chunks)}")
            except Exception as e:
                print(f"Worker crashed while processing chunk {i}/{len(chunks)}: {e}")
                print("Retrying this chunk in main process...")
                try:
                    # Retry in main process to avoid losing transcription
                    chunk_file, offset, _, _, _ = futures[i-1].args[0]  # recover args
                    segs = transcribe_single_chunk((chunk_file, offset, model_size, task, language))
                    all_segments.extend(segs)
                except Exception as e2:
                    print(f"Failed again for chunk {i}. Skipping. Error: {e2}")

    # Ensure results are in chronological order
    all_segments.sort(key=lambda x: x["start"])
    merged_segments = merge_overlapping_segments(all_segments, overlap=1.0)
    return {"segments": merged_segments, "language": language}

def generate_subtitles(video_path, subtitle_lang):
    """Main function to generate subtitles from video."""

    # 1. Extract audio
    audio_path, temp_dir = extract_audio(video_path)
    print(f"Extracted audio ...{audio_path}")


    # Find audio language
    chunks = split_audio(audio_path, chunk_length=600, overlap=1.0) 
    #     # --- Detect language from first chunk ---
    if chunks:
        first_chunk, offset = chunks[1] if len(chunks) > 1 else chunks[0]
        import stable_whisper
        model = stable_whisper.load_model("base")
        result = model.transcribe(first_chunk, task="transcribe", language=None)
        result = result.to_dict()
        audio_lang = result["language"]
        print(f"Detected language from first chunk: {audio_lang}")

    print(f"Detected audio language: {audio_lang}")
    print(f"Requested subtitle language: {subtitle_lang}")


    if audio_lang == subtitle_lang:
        # Case 2: same language → use transcribe task
        print("Using whisper transcribe task")
        transcript = transcribe_chunks_parallel(chunks, model_size="medium", task="transcribe", language=audio_lang)
        final_segments = transcript["segments"]

    else:
        # Case 1: Direct translation to English using Whisper
        print("Using Whisper translate task → English.")
        transcript = transcribe_chunks_parallel(chunks, model_size="medium", task="translate", language=audio_lang)
        segments = transcript["segments"]
        if subtitle_lang == "en":
            final_segments = segments
        else:
            # Case 3: different language and target != English
            print("Using M2M100 for translation.")
            if isinstance(segments, dict):
                # extract the actual segment list
                segments = segments.get("segments", [])
            texts = [seg["text"] for seg in segments]
            translated_texts = batch_translate_m2m100(texts, src_lang=audio_lang, tgt_lang=subtitle_lang)
            final_segments = []
            for seg, txt in zip(segments, translated_texts):
                final_segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": txt
                })
    
    shutil.rmtree(temp_dir, ignore_errors=True)
    return final_segments

def main():
    video_path = "new-data/A_discovery_of_witches.mp4"
    sub_format = "srt"
    final_segments = generate_subtitles(video_path, "en")

    subtitle_file = os.path.splitext(video_path)[0] + f".{sub_format}"
    write_subtitles(final_segments, sub_format, subtitle_file)
    print("Output: ", subtitle_file)

if __name__ == "__main__":
    main()
