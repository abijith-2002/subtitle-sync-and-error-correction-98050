import os
import time
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
import difflib
from typing import List, Dict, Any, Tuple
import re

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

    model_name = "m2m100_1.2B"
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

def _whisper_param_sweeps(task: str, language: str) -> List[Dict[str, Any]]:
    """
    Build a set of parameter combinations for multi-pass whisper runs.
    We vary temperature, beam_size/patience, and best_of to encourage diversity.
    """
    sweeps: List[Dict[str, Any]] = []
    temperatures = [0.0, 0.2, 0.4, 0.6]
    # For greedy vs beam search
    decoding_setups = [
        {"beam_size": None, "best_of": 5},         # greedy with best_of sampling headroom
        {"beam_size": 3, "best_of": None},
        {"beam_size": 5, "best_of": None},
    ]
    for t in temperatures:
        for d in decoding_setups:
            params = {
                "task": task,
                "language": language,
                "temperature": t,
                "beam_size": d["beam_size"],
                "best_of": d["best_of"],
                "patience": 1.0 if d["beam_size"] else None,
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6,
            }
            sweeps.append(params)
    return sweeps

def _normalize_seg_text(text: str) -> str:
    """Normalize text for voting/equality purposes."""
    t = re.sub(r"\s+", " ", text or "").strip().lower()
    return t

def _align_segment_groups(all_pass_segments: List[List[Dict[str, Any]]], time_tolerance: float = 0.6) -> List[List[Dict[str, Any]]]:
    """
    Align segments from multiple passes into groups that represent 'the same' time region.
    We use time overlap and proximity to group segments across passes.
    """
    # Flatten with pass index
    annotated = []
    for pass_idx, segs in enumerate(all_pass_segments):
        for s in segs:
            annotated.append((pass_idx, s))
    # Sort by start time
    annotated.sort(key=lambda x: x[1]["start"])
    groups: List[List[Dict[str, Any]]] = []
    current_group: List[Dict[str, Any]] = []

    def seg_close(a: Dict[str, Any], b: Dict[str, Any], tol: float) -> bool:
        # Consider overlap or close in time windows
        return (abs(a["start"] - b["start"]) <= tol) or (min(a["end"], b["end"]) - max(a["start"], b["start"]) > 0)

    for _, seg in annotated:
        if not current_group:
            current_group = [seg]
        else:
            if seg_close(current_group[-1], seg, time_tolerance):
                current_group.append(seg)
            else:
                groups.append(current_group)
                current_group = [seg]
    if current_group:
        groups.append(current_group)
    return groups

def _vote_texts(candidates: List[Dict[str, Any]]) -> Tuple[str, float]:
    """
    Simple ensemble voting:
    - Normalize candidate texts.
    - Count exact normalized matches.
    - If no majority, choose the text with highest average similarity to others.
    Returns chosen_text, agreement_ratio (0..1).
    """
    if not candidates:
        return "", 0.0
    texts = [c.get("text", "") for c in candidates]
    norm_texts = [_normalize_seg_text(t) for t in texts]
    # Exact match vote
    freq: Dict[str, int] = {}
    for nt in norm_texts:
        freq[nt] = freq.get(nt, 0) + 1
    best_norm, votes = max(freq.items(), key=lambda kv: kv[1])
    agreement_ratio = votes / max(1, len(norm_texts))

    # If we have a majority or at least 2/3 votes (tunable), use it
    if agreement_ratio >= 0.66:
        chosen = texts[norm_texts.index(best_norm)]
        return chosen, agreement_ratio

    # Otherwise, fall back to best average similarity
    def avg_similarity(idx: int) -> float:
        base = norm_texts[idx]
        sims = []
        for j, other in enumerate(norm_texts):
            if j == idx:
                continue
            sims.append(difflib.SequenceMatcher(None, base, other).ratio())
        return sum(sims) / max(1, len(sims))

    best_idx = max(range(len(norm_texts)), key=avg_similarity)
    # Estimate soft agreement via similarity to others
    soft_agreement = avg_similarity(best_idx)
    return texts[best_idx], soft_agreement

def _consolidate_groups(groups: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Build final segments from grouped candidates using voting and compute confidence per segment.
    Confidence is based on agreement (exact or similarity).
    """
    final: List[Dict[str, Any]] = []
    for group in groups:
        # determine representative timing: median start/end of group
        starts = sorted([g["start"] for g in group])
        ends = sorted([g["end"] for g in group])
        start = starts[len(starts)//2]
        end = ends[len(ends)//2]
        text, agreement = _vote_texts(group)
        confidence = float(agreement)  # 0..1
        low_confidence = confidence < 0.5  # threshold tunable

        final.append({
            "start": float(start),
            "end": float(end),
            "text": clean_text(text),
            "confidence": confidence,
            "needs_review": low_confidence
        })
    # order
    final.sort(key=lambda s: s["start"])
    # merge potential overlaps and keep higher confidence
    merged: List[Dict[str, Any]] = []
    for seg in final:
        if merged and seg["start"] < merged[-1]["end"]:
            # Overlap: keep the one with higher confidence or longer duration
            prev = merged[-1]
            score_prev = (prev.get("confidence", 0.0), prev["end"] - prev["start"])
            score_cur = (seg.get("confidence", 0.0), seg["end"] - seg["start"])
            if score_cur > score_prev:
                merged[-1] = seg
        else:
            merged.append(seg)
    return merged

def clean_text(text):
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces/newlines
    return text

def _transcribe_single_chunk_with_params(args):
    """
    Worker to transcribe a chunk with a specific parameter set for whisper.
    Returns list of segments with adjusted timestamps.
    """
    try:
        chunk_file, offset, model_size, params = args
        # load per process to avoid contention
        model = stable_whisper.load_model(model_size, device="cpu")
        # Filter out None values to avoid passing invalid args
        run_args = {k: v for k, v in params.items() if v is not None}
        result = model.transcribe(chunk_file, **run_args)
        result_dict = result.to_dict()

        if not isinstance(result_dict, dict) or "segments" not in result_dict:
            print(f"⚠️ No valid segments for chunk {chunk_file} with params {params}, skipping.")
            return []

        segs = result_dict["segments"]
        for seg in segs:
            seg["start"] += offset
            seg["end"] += offset
        return segs
    except Exception as e:
        print(f"⚠️ Error in chunk {chunk_file} with params {params}: {e}")
        return []

def transcribe_single_chunk(args):
    """
    Backwards-compatible single-pass transcriber used by older flow, preserved.
    """
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

def transcribe_chunks_parallel(chunks, model_size="large-v3", task="transcribe", language=None, enable_ensemble: bool = True):
    """
    Transcribe chunks in parallel. If enable_ensemble=True, run multi-pass parameter sweeps
    and perform ensemble voting + confidence scoring. Otherwise, preserve original single-pass behavior.
    """
    total_cpus = multiprocessing.cpu_count()
    cpu_load = psutil.cpu_percent(interval=0.1)
    if cpu_load > 70:  # system already busy
        max_workers = max(1, total_cpus // 4)  # be conservative
    else:
        max_workers = max(1, total_cpus // 2)

    print(f"Total chunks: {len(chunks)} (using {max_workers} workers on CPU)")

    if not enable_ensemble:
        # Original single-pass flow (preserved)
        all_segments = []
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
                        chunk_file, offset, _, _, _ = futures[i-1].args[0]  # recover args
                        segs = transcribe_single_chunk((chunk_file, offset, model_size, task, language))
                        all_segments.extend(segs)
                    except Exception as e2:
                        print(f"Failed again for chunk {i}. Skipping. Error: {e2}")

        all_segments.sort(key=lambda x: x["start"])
        merged_segments = merge_overlapping_segments(all_segments, overlap=1.0)
        return {"segments": merged_segments, "language": language}

    # New ensemble-enabled flow
    # 1) Prepare parameter sweeps
    sweeps = _whisper_param_sweeps(task=task, language=language)
    print(f"[Ensemble] Running {len(sweeps)} parameter configurations per chunk")

    # 2) For each chunk and sweep, run in parallel
    # We can run passes sequentially to control memory, but we'll parallelize across chunks.
    all_pass_results: List[List[Dict[str, Any]]] = []

    for pass_idx, params in enumerate(sweeps, start=1):
        print(f"[Ensemble] Pass {pass_idx}/{len(sweeps)} with params: {params}")
        pass_segments: List[Dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _transcribe_single_chunk_with_params,
                    (chunk_file, offset, model_size, params)
                )
                for chunk_file, offset in chunks
            ]
            for i, f in enumerate(as_completed(futures), start=1):
                try:
                    segs = f.result()
                    pass_segments.extend(segs)
                    print(f"[Ensemble] Finished chunk {i}/{len(chunks)} for pass {pass_idx}")
                except Exception as e:
                    print(f"[Ensemble] Worker crashed on pass {pass_idx}, chunk {i}/{len(chunks)}: {e}")
        # Sort and lightly merge per pass to reduce duplicates before global consolidation
        pass_segments.sort(key=lambda x: x["start"])
        pass_merged = merge_overlapping_segments(pass_segments, overlap=1.0)
        all_pass_results.append(pass_merged)

    # 3) Align groups across passes and vote
    groups = _align_segment_groups(all_pass_results, time_tolerance=0.6)
    consolidated = _consolidate_groups(groups)

    return {"segments": consolidated, "language": language}

def generate_subtitles(video_path, subtitle_lang):
    """Main function to generate subtitles from video."""

    # 1. Extract audio
    audio_path, temp_dir = extract_audio(video_path)
    print(f"Extracted audio ...{audio_path}")

    # Find audio language
    chunks = split_audio(audio_path, chunk_length=600, overlap=1.0) 
    # --- Detect language from first chunk ---
    if chunks:
        first_chunk, offset = chunks[1] if len(chunks) > 1 else chunks[0]
        model = stable_whisper.load_model("base")
        result = model.transcribe(first_chunk, task="transcribe", language=None)
        result = result.to_dict()
        audio_lang = result.get("language", None)
        print(f"Detected language from first chunk: {audio_lang}")

    print(f"Detected audio language: {audio_lang}")
    print(f"Requested subtitle language: {subtitle_lang}")

    if audio_lang == subtitle_lang:
        # Case 2: same language → use transcribe task
        print("Using whisper transcribe task with ensemble")
        transcript = transcribe_chunks_parallel(
            chunks,
            model_size="large-v3",
            task="transcribe",
            language=audio_lang,
            enable_ensemble=True  # enable ensemble by default
        )
        final_segments = transcript["segments"]

    else:
        # Case 1: Direct translation to English using Whisper
        print("Using Whisper translate task → English with ensemble.")
        transcript = transcribe_chunks_parallel(
            chunks,
            model_size="large-v3",
            task="translate",
            language=audio_lang,
            enable_ensemble=True
        )
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
                # Preserve confidence and needs_review flags if existed
                final_segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": txt,
                    "confidence": seg.get("confidence", None),
                    "needs_review": seg.get("needs_review", None)
                })
    
    shutil.rmtree(temp_dir, ignore_errors=True)
    return final_segments

def main():
    start = time.time()
    video_path = "Dangal.mp4"
    sub_format = "srt"
    print("starting")
    final_segments = generate_subtitles(video_path, "hi")

    subtitle_file = os.path.splitext(video_path)[0] + f".{sub_format}"
    write_subtitles(final_segments, sub_format, subtitle_file)
    print("Output: ", subtitle_file)
    end = time.time()
    print(f"total time taken : {end-start}")

if __name__ == "__main__":
    main()
