# Audio Transcriber

A local Python tool for transcribing interview audio and video recordings into text using `faster-whisper`.

This project is designed for researchers, students, and qualitative research workflows where recorded interviews need to be converted into readable and structured transcript files. It is especially useful for thesis interviews, semi-structured interviews, fieldwork recordings, and academic transcription workflows.

## Overview

This project allows the user to select an audio or video file from their computer and automatically generate transcript outputs.

The script opens a file selection window, loads a `faster-whisper` speech recognition model, transcribes the selected recording, divides the transcript into timestamped speech segments, and saves the results into multiple output formats.

By default, the project is configured for Turkish interview recordings and CPU-based execution.

## Features

- Local transcription without using a paid API
- Simple file selection through a desktop dialog
- Supports common audio and video formats such as `.mp3`, `.wav`, `.m4a`, `.aac`, `.flac`, `.ogg`, `.wma`, `.mp4`, `.mkv`, `.mov`, and `.avi`
- Generates clean TXT, timestamped TXT, SRT, and JSON outputs
- Includes timestamp information for each speech segment
- Uses voice activity detection to reduce silence and non-speech sections
- Works on CPU by default
- Configurable model size, language, device, compute type, beam size, and output folder name

## How It Works

The project uses the `faster-whisper` library to perform automatic speech recognition locally. After the user selects an audio or video file, the script loads the selected Whisper model and processes the recording segment by segment.

Each detected speech segment is stored with:

- segment ID
- start time
- end time
- transcribed text

The script then exports the transcript into four different formats:

- a clean text file for reading and academic writing,
- a timestamped text file for checking the transcript against the original recording,
- an SRT subtitle file,
- a JSON file containing structured transcript data and metadata.

## Installation

Python 3.10 is recommended.

Create and activate a new environment:

```bash
conda create -n voice_transcript python=3.10 -y
conda activate voice_transcript
```

Install the required package:

```bash
pip install faster-whisper
```

## Usage

Run the script:

```bash
python main.py
```

A file selection window will open. Select the interview audio or video file you want to transcribe.

After the transcription is completed, the output files will be saved into a folder named:

```text
transcript_output
```

This folder is created automatically next to the selected recording file.

## Output Files

For an input file named:

```text
interview.m4a
```

the script generates:

```text
interview_clean_transcript.txt
interview_timestamped_transcript.txt
interview_transcript.srt
interview_transcript.json
```

## Output File Descriptions

### Clean TXT

The clean transcript file contains only the transcribed text without timestamps. This file is useful for reading, editing, and transferring interview content into a thesis, report, article, or qualitative analysis document.

### Timestamped TXT

The timestamped transcript file includes the start and end time of each speech segment. This helps the user return to the original recording and verify specific statements.

### SRT

The SRT file stores the transcript in subtitle format. It can be used with media players, video editors, or other subtitle-compatible tools.

### JSON

The JSON file contains structured transcript data. It includes transcript segments, timestamps, text content, and metadata such as the selected model, detected language, elapsed transcription time, and number of detected segments.

## Configuration

The main settings can be changed inside the `TranscriptionConfig` class in `main.py`.

Default configuration:

```python
model_size = "small"
device = "cpu"
compute_type = "int8"
language = "tr"
beam_size = 5
vad_filter = True
output_folder_name = "transcript_output"
```

## Model Size Options

```text
tiny      fastest, lowest accuracy
base      fast, basic accuracy
small     good default option for CPU
medium    better accuracy, slower on CPU
large-v3  highest accuracy, very slow on CPU
```

The default setting is:

```python
model_size = "small"
```

For better accuracy, change it to:

```python
model_size = "medium"
```

For faster processing on weaker computers, use:

```python
model_size = "base"
```

or:

```python
model_size = "tiny"
```

## Language Settings

For Turkish recordings, use:

```python
language = "tr"
```

For English recordings, use:

```python
language = "en"
```

For automatic language detection, use:

```python
language = None
```

## CPU and GPU Usage

The script runs on CPU by default:

```python
device = "cpu"
compute_type = "int8"
```

If you have a CUDA-supported NVIDIA GPU, you can use:

```python
device = "cuda"
compute_type = "float16"
```
