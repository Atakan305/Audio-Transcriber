from faster_whisper import WhisperModel
from tkinter import Tk, filedialog
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
import time
import sys
import traceback

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class TranscriptionConfig:
    """
    Configuration settings for the transcription process.
    Change these values according to your system and transcript needs.
    """

    # Model options:
    # "tiny"     -> fastest, lowest accuracy
    # "base"     -> fast, basic accuracy
    # "small"    -> good first option for CPU
    # "medium"   -> better accuracy, slower on CPU
    # "large-v3" -> highest accuracy, very slow on CPU
    model_size: str = "small"

    # Device options:
    # "cpu"  -> works on most computers
    # "cuda" -> use this only if you have an NVIDIA GPU with CUDA properly installed
    device: str = "cpu"

    # Compute type:
    # "int8"    -> lower memory usage, good for CPU
    # "float16" -> good for GPU
    # "float32" -> more precise but heavier on CPU
    compute_type: str = "int8"

    # Language:
    # Turkish = "tr"
    # English = "en"
    # Automatic language detection = None
    language: Optional[str] = "tr"

    # Beam size:
    # Higher values may improve quality but slow down processing.
    beam_size: int = 5

    # Voice activity detection:
    # Helps ignore silence and non-speech parts.
    vad_filter: bool = True

    # Output folder name created next to the selected audio file.
    output_folder_name: str = "transcript_output"

# ============================================================
# FILE SELECTION
# ============================================================

def choose_audio_file() -> Optional[Path]:
    """
    Opens a file selection window and returns the selected audio/video file path.

    Returns
    -------
    Path or None
        Selected file path if the user chooses a file.
        None if the user cancels the selection.
    """

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(
        title="Select an audio or video file to transcribe",
        filetypes=[
            ("Audio files", "*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.wma"),
            ("Video files", "*.mp4 *.mkv *.mov *.avi"),
            ("All files", "*.*")
        ]
    )

    root.destroy()

    if not file_path:
        return None

    return Path(file_path)

# ============================================================
# TIME FORMATTING
# ============================================================

def format_srt_timestamp(seconds: float) -> str:
    """
    Converts seconds into SRT timestamp format.

    Example
    -------
    83.45 -> 00:01:23,450
    """

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)

    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def format_readable_timestamp(seconds: float) -> str:
    """
    Converts seconds into a readable timestamp format.

    Example
    -------
    83.45 -> 00:01:23
    """

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    return f"{hours:02}:{minutes:02}:{secs:02}"

# ============================================================
# OUTPUT WRITERS
# ============================================================

def write_txt_with_timestamps(
    output_path: Path,
    transcript_segments: List[Dict[str, Any]]
) -> None:
    """
    Writes a timestamped transcript file.
    """

    lines = []

    for segment in transcript_segments:
        start = format_readable_timestamp(segment["start"])
        end = format_readable_timestamp(segment["end"])
        text = segment["text"]

        lines.append(f"[{start} - {end}] {text}")

    output_path.write_text("\n".join(lines), encoding="utf-8")

def write_clean_txt(
    output_path: Path,
    transcript_segments: List[Dict[str, Any]]
) -> None:
    """
    Writes a clean transcript without timestamps.
    Useful for thesis writing and qualitative analysis.
    """

    text_blocks = [segment["text"] for segment in transcript_segments]
    output_path.write_text("\n\n".join(text_blocks), encoding="utf-8")

def write_srt(
    output_path: Path,
    transcript_segments: List[Dict[str, Any]]
) -> None:
    """
    Writes transcript in SRT subtitle format.
    """

    srt_blocks = []

    for segment in transcript_segments:
        index = segment["id"]
        start = format_srt_timestamp(segment["start"])
        end = format_srt_timestamp(segment["end"])
        text = segment["text"]

        block = f"{index}\n{start} --> {end}\n{text}\n"
        srt_blocks.append(block)

    output_path.write_text("\n".join(srt_blocks), encoding="utf-8")

def write_json(
    output_path: Path,
    transcript_segments: List[Dict[str, Any]],
    metadata: Dict[str, Any]
) -> None:
    """
    Writes detailed transcript data in JSON format.
    """

    data = {
        "metadata": metadata,
        "segments": transcript_segments
    }

    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

# ============================================================
# TRANSCRIPTION
# ============================================================

def load_model(config: TranscriptionConfig) -> WhisperModel:
    """
    Loads the faster-whisper model according to the selected configuration.
    """

    print("\nLoading speech recognition model...")
    print(f"Model size: {config.model_size}")
    print(f"Device: {config.device}")
    print(f"Compute type: {config.compute_type}")

    model = WhisperModel(
        config.model_size,
        device=config.device,
        compute_type=config.compute_type
    )

    return model

def transcribe_file(
    audio_path: Path,
    model: WhisperModel,
    config: TranscriptionConfig
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Transcribes the selected audio/video file and returns structured segments.

    Each segment includes:
    - segment id
    - start time
    - end time
    - transcript text
    """

    print("\nTranscription started.")
    print("For long interviews, this may take several minutes.")
    print("If Spyder looks frozen, do not close it immediately.\n")

    start_time = time.time()

    segments, info = model.transcribe(
        str(audio_path),
        language=config.language,
        beam_size=config.beam_size,
        vad_filter=config.vad_filter
    )

    transcript_segments = []

    for index, segment in enumerate(segments, start=1):
        text = segment.text.strip()

        if not text:
            continue

        segment_data = {
            "id": index,
            "start": float(segment.start),
            "end": float(segment.end),
            "text": text
        }

        transcript_segments.append(segment_data)

        start_label = format_readable_timestamp(segment.start)
        end_label = format_readable_timestamp(segment.end)

        print(f"[{start_label} - {end_label}] {text}")

    elapsed_time = time.time() - start_time

    metadata = {
        "audio_file": str(audio_path),
        "model_size": config.model_size,
        "device": config.device,
        "compute_type": config.compute_type,
        "language_setting": config.language,
        "detected_language": info.language,
        "language_probability": float(info.language_probability),
        "elapsed_minutes": round(elapsed_time / 60, 2),
        "number_of_segments": len(transcript_segments)
    }

    return transcript_segments, metadata

# ============================================================
# MAIN PROGRAM
# ============================================================

def main() -> None:
    """
    Main execution function.
    """

    config = TranscriptionConfig()

    print("=" * 60)
    print("INTERVIEW AUDIO TRANSCRIBER")
    print("=" * 60)
    print("This program converts interview audio/video files into text.")
    print("A file selection window will open now.")

    audio_path = choose_audio_file()

    if audio_path is None:
        print("\nNo file selected. Program stopped.")
        return

    if not audio_path.exists():
        print("\nError: Selected file does not exist.")
        return

    print("\nSelected file:")
    print(audio_path)

    output_folder = audio_path.parent / config.output_folder_name
    output_folder.mkdir(exist_ok=True)

    print("\nOutput folder:")
    print(output_folder)

    base_name = audio_path.stem

    timestamped_txt_output = output_folder / f"{base_name}_timestamped_transcript.txt"
    clean_txt_output = output_folder / f"{base_name}_clean_transcript.txt"
    srt_output = output_folder / f"{base_name}_transcript.srt"
    json_output = output_folder / f"{base_name}_transcript.json"

    try:
        model = load_model(config)

        transcript_segments, metadata = transcribe_file(
            audio_path=audio_path,
            model=model,
            config=config
        )

        if not transcript_segments:
            print("\nNo speech segments were detected.")
            return

        write_txt_with_timestamps(timestamped_txt_output, transcript_segments)
        write_clean_txt(clean_txt_output, transcript_segments)
        write_srt(srt_output, transcript_segments)
        write_json(json_output, transcript_segments, metadata)

        print("\n" + "=" * 60)
        print("TRANSCRIPTION COMPLETED SUCCESSFULLY")
        print("=" * 60)

        print(f"\nDetected language: {metadata['detected_language']}")
        print(f"Language probability: {metadata['language_probability']:.2f}")
        print(f"Total segments: {metadata['number_of_segments']}")
        print(f"Elapsed time: {metadata['elapsed_minutes']} minutes")

        print("\nGenerated files:")
        print(f"1. Timestamped TXT: {timestamped_txt_output}")
        print(f"2. Clean TXT:       {clean_txt_output}")
        print(f"3. SRT subtitle:    {srt_output}")
        print(f"4. JSON data:       {json_output}")

        print("\nYou can use the clean TXT file for thesis writing.")
        print("You can use the timestamped TXT, SRT, or JSON files for verification.")

    except Exception as error:
        print("\n" + "=" * 60)
        print("ERROR DURING TRANSCRIPTION")
        print("=" * 60)
        print(f"\nError type: {type(error).__name__}")
        print(f"Error message: {error}")

        print("\nFull traceback:")
        traceback.print_exc()

        print("\nPossible solutions:")
        print("1. Make sure faster-whisper is installed in the active environment.")
        print("2. Make sure the selected file is a valid audio or video file.")
        print("3. If model download fails on Windows, run Spyder as administrator.")
        print("4. If the process is too slow, use model_size='tiny' or model_size='base'.")
        print("5. If you have a CUDA-supported GPU, set device='cuda' and compute_type='float16'.")

if __name__ == "__main__":
    main() 
