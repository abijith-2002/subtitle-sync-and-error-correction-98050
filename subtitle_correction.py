import os
import re
import difflib
from typing import List, Dict, Any, Tuple

import stable_whisper
from sub_gen_refined import *  # uses audio extraction, chunking, and baseline transcribe utilities

# PUBLIC_INTERFACE
def ensemble_multipass_transcription(
    video_path: str,
    language_hint: str = None,
    model_size: str = "medium",
    run_params: List[Dict[str, Any]] = None,
    confidence_threshold: float = 0.6,
) -> Dict[str, Any]:
    """Run Whisper multiple times with different decoding parameters and ensemble their outputs.

    PUBLIC_INTERFACE
    This is a public function.

    Args:
        video_path: Path to the video file.
        language_hint: Optional language hint passed to Whisper.
        model_size: Whisper model size to load for each pass.
        run_params: List of decoding parameter dicts for each pass. If None, sensible defaults are used.
            Example for each dict:
            {
                "temperature": 0.0,
                "beam_size": 5,
                "patience": 1.0,
                "best_of": 1,
                "task": "transcribe" or "translate"
            }
        confidence_threshold: Threshold in [0,1]. Segments below this agreement will be flagged low-confidence.

    Returns:
        dict with keys:
            - "segments": list of ensembled segments with fields:
                start, end, text, confidence (0..1), low_confidence (bool)
            - "language": detected/used language
            - "meta": details about runs and parameters
    """
    # Default run parameters to explore decoding space
    default_runs = [
        {"temperature": 0.0, "beam_size": 5, "patience": 1.0, "best_of": 1},
        {"temperature": 0.2, "beam_size": 5, "patience": 1.0, "best_of": 1},
        {"temperature": 0.5, "beam_size": 3, "patience": 1.0, "best_of": 2},
        {"temperature": 0.8, "beam_size": 1, "patience": 1.0, "best_of": 3},
    ]
    run_params = run_params or default_runs

    # 1) Extract audio and split into chunks using existing utilities
    audio_path, temp_dir = extract_audio(video_path)
    chunks = split_audio(audio_path, chunk_length=600, overlap=1.0)

    # 2) Language detection from the first chunk if not given
    audio_lang = language_hint
    try:
        if not audio_lang and chunks:
            first_chunk, _ = chunks[1] if len(chunks) > 1 else chunks[0]
            model_probe = stable_whisper.load_model("base")
            probe_result = model_probe.transcribe(first_chunk, task="transcribe", language=None).to_dict()
            audio_lang = probe_result.get("language", None)
            print(f"[Ensemble] Detected language: {audio_lang}")
    except Exception as e:
        print(f"[Ensemble] Language detection failed: {e}")

    # 3) Run multiple passes varying decoding parameters
    all_runs_segments: List[List[Dict[str, Any]]] = []
    meta_runs = []
    for i, params in enumerate(run_params, start=1):
        task_mode = params.get("task", "transcribe")

        print(f"[Ensemble] Pass {i}/{len(run_params)} with params: {params} | task={task_mode}")

        try:
            # Reuse the parallel chunk transcribe from sub_gen_refined, passing language/task through.
            transcript = transcribe_chunks_parallel(
                chunks,
                model_size=model_size,
                task=task_mode,
                language=audio_lang,
            )
            segs = transcript.get("segments", [])
            # Normalize minimal fields
            normalized = []
            for s in segs:
                if not isinstance(s, dict):
                    continue
                text = s.get("text", "").strip()
                if not text:
                    continue
                normalized.append({
                    "start": float(s.get("start", 0.0)),
                    "end": float(s.get("end", 0.0)),
                    "text": text,
                })
            all_runs_segments.append(normalized)
            meta_runs.append({"pass": i, "params": params, "segments": len(normalized)})
            print(f"[Ensemble] Pass {i} produced {len(normalized)} segments.")
        except Exception as e:
            print(f"[Ensemble] Pass {i} failed: {e}")
            all_runs_segments.append([])
            meta_runs.append({"pass": i, "params": params, "segments": 0, "error": str(e)})

    # 4) Ensemble voting across runs
    ensembled_segments = _ensemble_segments_voting(
        all_runs_segments,
        agreement_window=0.5,   # seconds tolerance for start/end alignment across runs
        text_similarity=0.7,    # ratio threshold for texts to be considered matching
    )

    # 5) Confidence scoring
    N = max(1, len(all_runs_segments))
    for seg in ensembled_segments:
        votes = seg.get("votes", 1)
        seg["confidence"] = round(votes / N, 3)
        seg["low_confidence"] = seg["confidence"] < confidence_threshold
        seg.pop("votes", None)

    # 6) Cleanup temp dir
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    return {
        "segments": ensembled_segments,
        "language": audio_lang,
        "meta": {"runs": meta_runs, "num_runs": len(all_runs_segments)},
    }


def _text_sim(a: str, b: str) -> float:
    """Compute similarity score between two strings."""
    return difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _time_overlap_or_close(a: Tuple[float, float], b: Tuple[float, float], tol: float) -> bool:
    """True if [a] and [b] are sufficiently close in start/end within tol seconds."""
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def _ensemble_segments_voting(
    runs_segments: List[List[Dict[str, Any]]],
    agreement_window: float = 0.5,
    text_similarity: float = 0.7,
) -> List[Dict[str, Any]]:
    """Combine multiple transcription runs into a single list via voting.

    Strategy:
    - Create candidate "bins" from the first successful run (or progressively).
    - For each segment in other runs, try to match into an existing bin if
      start/end are within agreement_window and text similarity >= threshold.
    - If no bin matches, create a new bin.
    - Each bin collects votes, and representative text/time is chosen by median time and the most frequent text.

    Returns:
        List of segments: {start, end, text, votes}
    """
    bins: List[Dict[str, Any]] = []

    # Helper: add or vote a segment into bins
    def add_vote(seg: Dict[str, Any]):
        nonlocal bins
        for b in bins:
            if _time_overlap_or_close((b["start"], b["end"]), (seg["start"], seg["end"]), agreement_window):
                # Compare to representative text
                if _text_sim(b["text"], seg["text"]) >= text_similarity:
                    b["votes"] += 1
                    b["texts"].append(seg["text"])
                    b["starts"].append(seg["start"])
                    b["ends"].append(seg["end"])
                    return True
        return False

    # Initialize bins with the first non-empty run if available
    seeded = False
    for run in runs_segments:
        if run and not seeded:
            for s in run:
                bins.append({
                    "start": s["start"],
                    "end": s["end"],
                    "text": s["text"],
                    "votes": 1,
                    "texts": [s["text"]],
                    "starts": [s["start"]],
                    "ends": [s["end"]],
                })
            seeded = True
            break

    # If all runs empty
    if not bins:
        return []

    # Process the remaining runs
    skip_first = True
    for run in runs_segments:
        if skip_first and seeded:
            skip_first = False
            continue
        for s in run:
            matched = add_vote(s)
            if not matched:
                # Create a new bin if no match
                bins.append({
                    "start": s["start"],
                    "end": s["end"],
                    "text": s["text"],
                    "votes": 1,
                    "texts": [s["text"]],
                    "starts": [s["start"]],
                    "ends": [s["end"]],
                })

    # Finalize bins by consolidating text and using robust time statistics (median)
    import statistics
    result = []
    for b in bins:
        # choose the most frequent text (ties broken by longest text)
        text_freq: Dict[str, int] = {}
        for t in b["texts"]:
            text_freq[t] = text_freq.get(t, 0) + 1
        best_text = max(text_freq.items(), key=lambda kv: (kv[1], len(kv[0])))[0]

        # median times
        try:
            start_med = statistics.median(b["starts"])
            end_med = statistics.median(b["ends"])
        except statistics.StatisticsError:
            start_med = b["start"]
            end_med = b["end"]

        result.append({
            "start": float(start_med),
            "end": float(end_med),
            "text": best_text,
            "votes": b["votes"],
        })

    # Sort by start time
    result.sort(key=lambda x: x["start"])
    return result


class SubtitleSynchronizer:
    def __init__(self, video_path, subtitle_path, output_path=None):
        self.video_path = video_path
        self.subtitle_path = subtitle_path
        self.output_path = output_path or self._generate_output_path()
        self.audio_sample_rate = 22050
        self.hop_length = 512
        self.frame_length = 2048
        # Optional model handle (not strictly needed with ensemble)
        try:
            self.model = stable_whisper.load_model("medium")
        except Exception:
            self.model = None
        
    def _generate_output_path(self):
        """Generate output path with _synced suffix"""
        base, ext = os.path.splitext(self.subtitle_path)
        return f"{base}_synced{ext}"
    
    def parse_subtitle_file(self):
        """Parse subtitle file and detect format"""
        with open(self.subtitle_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if content.strip().startswith('WEBVTT'):
            return self._parse_vtt(content)
        elif re.search(r'\d+\n\d{2}:\d{2}:\d{2},\d{3}', content):
            return self._parse_srt(content)
        elif re.search(r'<\?xml[^>]*\?>\s*<\s*tt\s+[^>]*xmlns\s*=\s*["\']http://www.w3.org/ns/ttml["\'][^>]*>', content, re.IGNORECASE):
            return self._parse_ttml(content)
        else:
            raise ValueError("Unsupported subtitle format")
    
    def _parse_srt(self, content):
        """Parse SRT subtitle format"""
        subtitles = []
        blocks = re.split(r'\n\s*\n', content.strip())
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                try:
                    index = int(lines[0])
                    time_line = lines[1]
                    text = '\n'.join(lines[2:])
                    
                    time_match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})', time_line)
                    if time_match:
                        start_time = self._time_to_seconds(*time_match.groups()[:4])
                        end_time = self._time_to_seconds(*time_match.groups()[4:])
                        
                        subtitles.append({
                            'index': index,
                            'start': start_time,
                            'end': end_time,
                            'text': text,
                            'format': 'srt'
                        })
                except (ValueError, AttributeError):
                    continue
        
        return subtitles
    
    def _parse_vtt(self, content):
        """Parse WebVTT subtitle format"""
        subtitles = []
        lines = content.split('\n')
        i = 0
        index = 1
        
        while i < len(lines):
            line = lines[i].strip()
            time_match = re.match(r'(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})', line)
            if time_match:
                start_time = self._time_to_seconds(*time_match.groups()[:4])
                end_time = self._time_to_seconds(*time_match.groups()[4:])
                
                text_lines = []
                i += 1
                while i < len(lines) and lines[i].strip():
                    text_lines.append(lines[i].strip())
                    i += 1
                
                if text_lines:
                    subtitles.append({
                        'index': index,
                        'start': start_time,
                        'end': end_time,
                        'text': '\n'.join(text_lines),
                        'format': 'vtt'
                    })
                    index += 1
            
            i += 1
        
        return subtitles
    
    def _parse_ttml(self, content):
        pattern = r'<p[^>]*begin="(\d{2}):(\d{2}):(\d{2})\.(\d{3})"\s+end="(\d{2}):(\d{2}):(\d{2})\.(\d{3})"[^>]*>(.*?)</p>'
        matches = re.findall(pattern, content, re.DOTALL)

        subtitles = []
        for idx, match in enumerate(matches, 1):
            begin_hours, begin_min, begin_sec, begin_millisec = map(int, match[0:4])
            end_hours, end_min, end_sec, end_millisec = map(int, match[4:8])
            text = match[8].strip().replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
            
            start_seconds = self._time_to_seconds(begin_hours, begin_min, begin_sec, begin_millisec)
            end_seconds = self._time_to_seconds(end_hours, end_min, end_sec, end_millisec)
            
            subtitle = {
                'index': idx,
                'start': start_seconds,
                'end': end_seconds,
                'text': text,
                'format': 'ttml'
            }
            subtitles.append(subtitle)
        
        return subtitles

    def _parse_ass(self, content):
        """Parse ASS file content and return a list of subtitle dictionaries.
        
        Args:
            content (str): String containing ASS file content.
        
        Returns:
            list: List of dictionaries with keys 'index', 'start', 'end', 'text', 'format'.
        """
        if not re.search(r'\[Events\]', content, re.IGNORECASE):
            raise ValueError("Invalid ASS content: [Events] section missing")

        pattern = r'^Dialogue:\s*[^,]*,\s*(\d+:\d{2}:\d{2}\.\d{2}),\s*(\d+:\d{2}:\d{2}\.\d{2}),\s*[^,]*,\s*[^,]*,\s*[^,]*,\s*[^,]*,\s*[^,]*,\s*[^,]*,\s*(.*)$'
        matches = re.findall(pattern, content, re.MULTILINE)

        subtitles = []
        for index, match in enumerate(matches, 1):
            start_timestamp = match[0]
            start_parts = start_timestamp.replace(':', '.').split('.')
            start_hours, start_minutes, start_seconds, start_centiseconds = map(int, start_parts)
            start_time = self._time_to_seconds(start_hours, start_minutes, start_seconds, start_centiseconds * 10)

            end_timestamp = match[1]
            end_parts = end_timestamp.replace(':', '.').split('.')
            end_hours, end_minutes, end_seconds, end_centiseconds = map(int, end_parts)
            end_time = self._time_to_seconds(end_hours, end_minutes, end_seconds, end_centiseconds * 10)

            text = match[2].strip().replace('\\N', '\n')

            subtitles.append({
                'index': index,
                'start': start_time,
                'end': end_time,
                'text': text,
                'format': 'ass'
            })

        return subtitles
    
    def _time_to_seconds(self, hours, minutes, seconds, milliseconds):
        """Convert time components to total seconds"""
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000
    

    def synchronize(self, confidence_threshold: float = 0.6):
        """Main synchronization with Multi-pass Ensemble + Confidence scoring.

        Steps:
        - Parse existing subtitle file to get base segments and their language.
        - Run multi-pass Whisper over the audio and ensemble the results.
        - Compute confidence per segment and flag low-confidence ones.
        - Align the ensembled transcript to the provided subtitles.
        """
        try:
            print("Parsing subtitles...")
            subtitles = self.parse_subtitle_file()
            if not subtitles:
                raise ValueError("No valid subtitles found")
            
            sub_lang = detectLang(subtitles)
            print("Detected language from the subtitle: ", sub_lang)

            print("Running ensemble multi-pass transcription...")
            ensemble_out = ensemble_multipass_transcription(
                self.video_path,
                language_hint=sub_lang,
                model_size="medium",
                run_params=None,  # use defaults
                confidence_threshold=confidence_threshold,
            )
            transcript = ensemble_out.get("segments", [])
            if not transcript:
                print("Warning: There was an error or empty result during transcription ensemble.")
                return False

            # Align ensembled transcript to existing subtitles
            aligned = align_subtitles(transcript, subtitles)
            print("Subtitle correction and alignment done.")

            # Optionally log low-confidence indexes for human review
            low_conf_indices = [i for i, seg in enumerate(transcript) if seg.get("low_confidence")]
            if low_conf_indices:
                print(f"[Review] Low-confidence segments detected: {len(low_conf_indices)}")
            else:
                print("[Review] No low-confidence segments detected.")

            self.write_subtitle_file(aligned, self.output_path)
            print(f"Corrected subtitles saved to: {self.output_path}")
        except Exception as e:
            print("Error: ", e)
    
    def write_subtitle_file(self, subtitles, output_path):
        """Write subtitles to file in original format"""
        if not subtitles:
            return
        
        format_type = subtitles[0]['format']

        if format_type == 'vtt':
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("WEBVTT\n\n")
                for sub in subtitles:
                    start_time = self._seconds_to_vtt_time(sub['start'])
                    end_time = self._seconds_to_vtt_time(sub['end'])
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{sub['text']}\n\n")
            
        elif format_type == 'srt':
            with open(output_path, 'w', encoding='utf-8') as f:
                for sub in subtitles:
                    start_time = self._seconds_to_srt_time(sub['start'])
                    end_time = self._seconds_to_srt_time(sub['end'])
                    f.write(f"{sub['index']}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{sub['text']}\n\n")

        elif format_type == 'ttml':
            with open(self.subtitle_path, 'r', encoding='utf-8') as f:
                content = f.read()
            header_match = re.search(r'^(.*?<\s*/head\s*>)', content, re.DOTALL | re.IGNORECASE)
            header = header_match.group(1)

            p_match = re.search(r'<p\s+([^>]*begin="[^"]*"[^>]*end="[^"]*"[^>]*)>(.*?)</p>', content, re.DOTALL)
            style_attr = 'style="defaultStyle"'
            region_attr = 'region="bottom"'
            if p_match:
                attrs = p_match.group(1)
                style_match = re.search(r'style\s*=\s*["\']([^"\']*)["\']', attrs)
                region_match = re.search(r'region\s*=\s*["\']([^"\']*)["\']', attrs)
                if style_match:
                    style_attr = f'style="{style_match.group(1)}"'
                if region_match:
                    region_attr = f'region="{region_match.group(1)}"'
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(header + '\n')
                f.write('  <body>\n')
                f.write('    <div>\n')
                for sub in subtitles:
                    start_time = self._seconds_to_ttml_time(sub['start'])
                    end_time = self._seconds_to_ttml_time(sub['end'])
                    text = sub['text'].strip().replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    f.write(
                        f'      <p {style_attr} {region_attr} begin="{start_time}" '
                        f'end="{end_time}">{text}</p>\n'
                    )
                f.write('    </div>\n')
                f.write('  </body>\n')
                f.write('</tt>\n')

    
    def _seconds_to_srt_time(self, seconds):
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def _seconds_to_vtt_time(self, seconds):
        """Convert seconds to VTT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"
    
    def _seconds_to_ttml_time(self, seconds):        
        """Convert seconds to TTML timestamp format (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
    

# Helper functions to check subtitle text language 
from langdetect import detect
def detectLang(sub):
    text = ''
    for i in range(min(10, len(sub))):
        if sub[i]['text']:
            text = text + sub[i]['text'] + ' '
    lang = detect(text) if text.strip() else "en"
    return lang


def word_recall_ratio(sub_text, transcript_text):
    """Compute recall ratio: fraction of transcript words present in subtitle."""
    sub_words = set(sub_text.lower().split())
    t_words = transcript_text.lower().split()
    if not t_words:
        return 1.0
    common = sum(1 for w in t_words if w in sub_words)
    return common / len(t_words)

def align_subtitles(transcript, subtitles, time_tolerance=0.8, sim_threshold=0.7):
    """
    Align subtitles to transcript:
    - Keeps subtitle text if meaning is same (small differences).
    - Replaces with transcript text if subtitle is clearly wrong.
    - Inserts missing transcript lines.
    - Drops extra subtitle lines.
    - Works for all languages (monolingual).
    """
    corrected = []
    t_index, s_index = 0, 0
    used_transcript = set()  # to avoid duplicate insertions

    while t_index < len(transcript) and s_index < len(subtitles):
        t_seg = transcript[t_index]
        s_seg = subtitles[s_index]

        t_start, t_end, t_text = t_seg["start"], t_seg["end"], t_seg["text"].strip()
        s_start, s_end, s_text = s_seg["start"], s_seg["end"], s_seg["text"].strip()

        time_close = abs(t_start - s_start) <= time_tolerance and abs(t_end - s_end) <= time_tolerance
        sim = difflib.SequenceMatcher(None, t_text.lower(), s_text.lower()).ratio()

        if time_close:
            if sim >= sim_threshold:
                recall = word_recall_ratio(s_text, t_text)
                if recall < 0.9:
                    corrected.append({
                        "index": len(corrected) + 1,
                        "start": t_start,
                        "end": t_end,
                        "text": t_text,
                        "format": s_seg.get("format", "srt")
                    })
                    print(f"[REPLACE-MISSING] Subtitle {s_seg.get('index','?')}: missing words (recall={recall:.2f})\n")
                else:
                    corrected.append({
                        "index": len(corrected) + 1,
                        "start": t_start,
                        "end": t_end,
                        "text": s_text,
                        "format": s_seg.get("format", "srt")
                    })
                    print(f"[KEEP] Subtitle {s_seg.get('index','?')}: kept '{s_text}' (sim={sim:.2f})")
            else:
                corrected.append({
                    "index": len(corrected) + 1,
                    "start": t_start,
                    "end": t_end,
                    "text": t_text,
                    "format": s_seg.get("format", "srt")
                })
                print(f"[REPLACE] Subtitle {s_seg.get('index','?')}: '{s_text}' → '{t_text}' (sim={sim:.2f})")
            t_index += 1
            s_index += 1

        elif t_end < s_start - time_tolerance:
            if t_index not in used_transcript:
                corrected.append({
                    "index": len(corrected) + 1,
                    "start": t_start,
                    "end": t_end,
                    "text": t_text,
                    "format": "srt"
                })
                print(f"[INSERT] From transcript: '{t_text}' ({t_start:.2f}-{t_end:.2f})")
                used_transcript.add(t_index)
            t_index += 1

        elif s_end < t_start - time_tolerance:
            print(f"[DROP] Subtitle {s_seg.get('index','?')}: '{s_text}' ({s_start:.2f}-{s_end:.2f}) not in transcript")
            s_index += 1

        else:
            corrected.append({
                "index": len(corrected) + 1,
                "start": t_start,
                "end": t_end,
                "text": t_text,
                "format": s_seg.get("format", "srt")
            })
            print(f"[ALIGN-FORCE] Subtitle {s_seg.get('index','?')} forced to transcript '{t_text}'")
            t_index += 1
            s_index += 1

    while t_index < len(transcript):
        if t_index not in used_transcript:
            t_seg = transcript[t_index]
            corrected.append({
                "index": len(corrected) + 1,
                "start": t_seg["start"],
                "end": t_seg["end"],
                "text": t_seg["text"].strip(),
                "format": "srt"
            })
            print(f"[INSERT-TAIL] From transcript: '{t_seg['text']}' ({t_seg['start']:.2f}-{t_seg['end']:.2f})")
        t_index += 1

    return corrected


def main():
    video_file = "test-data/2mins.mp4"
    subtitle_file = "test-data/2mins.srt"

    synchronizer = SubtitleSynchronizer(video_file, subtitle_file)
    synchronizer.synchronize(confidence_threshold=0.6)

if __name__ == "__main__":
    print("Program started ✅")
    main()
