from typing import Any, List, Optional, Union, Callable
from dataclasses import dataclass
import base64
import datetime
import subprocess
import os
import requests
import time
import torch
import re

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import torchaudio


@dataclass
class Output:
    segments: list
    language: Optional[str] = None
    num_speakers: Optional[int] = None


class Predictor:
    def __init__(self, hf_token: str = None):
        self.hf_token = hf_token
        self.model = None
        self.diarization_model = None
        self.progress_callback = None

    def setup(self, hf_token: Optional[str] = None, progress_callback: Optional[Callable] = None):
        """Load the model into memory to make running multiple predictions efficient"""
        if progress_callback:
            self.progress_callback = progress_callback
        
        if hf_token:
            self.hf_token = hf_token
        if not self.hf_token:
            raise ValueError("HuggingFace token is required")

        self._update_progress("正在加载 Whisper 模型...", 0)
        model_name = "large-v3"
        self.model = WhisperModel(
            model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16",
        )
        
        self._update_progress("正在加载说话人分离模型...", 10)
        self.diarization_model = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.hf_token,
        ).to(torch.device("cuda"))
        
        self._update_progress("模型加载完成", 20)

    def _update_progress(self, status: str, progress: int):
        """更新进度信息"""
        if self.progress_callback:
            self.progress_callback({"status": status, "progress": progress})

    def predict(
        self,
        file_string: Optional[str] = None,
        file_url: Optional[str] = None,
        file: Optional[str] = None,
        group_segments: bool = True,
        transcript_output_format: str = "both",
        num_speakers: Optional[int] = None,
        translate: bool = False,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        offset_seconds: int = 0,
    ) -> Output:
        try:
            self._update_progress("开始处理音频文件...", 25)
            temp_wav_filename = f"temp-{time.time_ns()}.wav"

            if file is not None:
                self._update_progress("正在转换音频格式...", 30)
                subprocess.run(
                    [
                        "ffmpeg",
                        "-i",
                        file,
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        "-c:a",
                        "pcm_s16le",
                        temp_wav_filename,
                    ]
                )

            elif file_url is not None:
                self._update_progress("正在下载音频文件...", 30)
                response = requests.get(file_url)
                temp_audio_filename = f"temp-{time.time_ns()}.audio"
                with open(temp_audio_filename, "wb") as file:
                    file.write(response.content)

                subprocess.run(
                    [
                        "ffmpeg",
                        "-i",
                        temp_audio_filename,
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        "-c:a",
                        "pcm_s16le",
                        temp_wav_filename,
                    ]
                )

                if os.path.exists(temp_audio_filename):
                    os.remove(temp_audio_filename)
            elif file_string is not None:
                self._update_progress("正在解码音频数据...", 30)
                audio_data = base64.b64decode(
                    file_string.split(",")[1] if "," in file_string else file_string
                )
                temp_audio_filename = f"temp-{time.time_ns()}.audio"
                with open(temp_audio_filename, "wb") as f:
                    f.write(audio_data)

                subprocess.run(
                    [
                        "ffmpeg",
                        "-i",
                        temp_audio_filename,
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        "-c:a",
                        "pcm_s16le",
                        temp_wav_filename,
                    ]
                )

                if os.path.exists(temp_audio_filename):
                    os.remove(temp_audio_filename)

            self._update_progress("开始转录音频...", 40)
            segments, detected_num_speakers, detected_language = self.speech_to_text(
                temp_wav_filename,
                num_speakers,
                prompt,
                offset_seconds,
                group_segments,
                language,
                word_timestamps=True,
                transcript_output_format=transcript_output_format,
                translate=translate,
            )

            self._update_progress("处理完成", 100)
            return Output(
                segments=segments,
                language=detected_language,
                num_speakers=detected_num_speakers,
            )

        except Exception as e:
            self._update_progress(f"处理出错: {str(e)}", -1)
            raise RuntimeError("Error Running inference with local model", e)

        finally:
            if os.path.exists(temp_wav_filename):
                os.remove(temp_wav_filename)

    def convert_time(self, secs, offset_seconds=0):
        return datetime.timedelta(seconds=(round(secs) + offset_seconds))

    def speech_to_text(
        self,
        audio_file_wav,
        num_speakers=None,
        prompt="",
        offset_seconds=0,
        group_segments=True,
        language=None,
        word_timestamps=True,
        transcript_output_format="both",
        translate=False,
    ):
        self._update_progress("开始语音识别...", 50)
        time_start = time.time()

        # Transcribe audio
        print("Starting transcribing")
        options = dict(
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=1000),
            initial_prompt=prompt,
            word_timestamps=word_timestamps,
            language=language,
            task="translate" if translate else "transcribe",
            hotwords=prompt
        )
        segments, transcript_info = self.model.transcribe(audio_file_wav, **options)
        segments = list(segments)
        segments = [
            {
                "avg_logprob": s.avg_logprob,
                "start": float(s.start + offset_seconds),
                "end": float(s.end + offset_seconds),
                "text": s.text,
                "words": [
                    {
                        "start": float(w.start + offset_seconds),
                        "end": float(w.end + offset_seconds),
                        "word": w.word,
                        "probability": w.probability,
                    }
                    for w in s.words
                ],
            }
            for s in segments
        ]

        time_transcribing_end = time.time()
        print(
            f"Finished with transcribing, took {time_transcribing_end - time_start:.5} seconds, {len(segments)} segments"
        )

        print("Starting diarization")
        waveform, sample_rate = torchaudio.load(audio_file_wav)
        self._update_progress("开始说话人分离...", 60)
        diarization = self.diarization_model(
            {"waveform": waveform, "sample_rate": sample_rate},
            num_speakers=num_speakers,
        )

        time_diraization_end = time.time()
        print(
            f"Finished with diarization, took {time_diraization_end - time_transcribing_end:.5} seconds"
        )

        print("Starting merging")

        # Initialize variables to keep track of the current position in both lists
        margin = 0.1  # 0.1 seconds margin

        # Initialize an empty list to hold the final segments with speaker info
        final_segments = []

        diarization_list = list(diarization.itertracks(yield_label=True))
        unique_speakers = {
            speaker for _, _, speaker in diarization.itertracks(yield_label=True)
        }
        detected_num_speakers = len(unique_speakers)

        speaker_idx = 0
        n_speakers = len(diarization_list)

        # Iterate over each segment
        for segment in segments:
            segment_start = segment["start"] + offset_seconds
            segment_end = segment["end"] + offset_seconds
            segment_text = []
            segment_words = []

            # Iterate over each word in the segment
            for word in segment["words"]:
                word_start = word["start"] + offset_seconds - margin
                word_end = word["end"] + offset_seconds + margin

                while speaker_idx < n_speakers:
                    turn, _, speaker = diarization_list[speaker_idx]

                    if turn.start <= word_end and turn.end >= word_start:
                        # Add word without modifications
                        segment_text.append(word["word"])

                        # Strip here for individual word storage
                        word["word"] = word["word"].strip()
                        segment_words.append(word)

                        if turn.end <= word_end:
                            speaker_idx += 1

                        break
                    elif turn.end < word_start:
                        speaker_idx += 1
                    else:
                        break

            if segment_text:
                combined_text = "".join(segment_text)
                cleaned_text = re.sub("  ", " ", combined_text).strip()
                new_segment = {
                    "avg_logprob": segment["avg_logprob"],
                    "start": segment_start - offset_seconds,
                    "end": segment_end - offset_seconds,
                    "speaker": speaker,
                    "text": cleaned_text,
                    "words": segment_words,
                }
                final_segments.append(new_segment)

        time_merging_end = time.time()
        print(
            f"Finished with merging, took {time_merging_end - time_diraization_end:.5} seconds, {len(final_segments)} segments, {detected_num_speakers} speakers"
        )

        # Check if final_segments is empty
        if not final_segments:
            return [], detected_num_speakers, transcript_info.language

        print("Starting cleaning")
        segments = final_segments
        # Make output
        output = []  # Initialize an empty list for the output

        # Initialize the first group with the first segment
        current_group = {
            "start": segments[0]["start"],
            "end": segments[0]["end"],
            "speaker": segments[0]["speaker"],
            "avg_logprob": segments[0]["avg_logprob"],
        }

        if transcript_output_format in ("segments_only", "both"):
            current_group["text"] = segments[0]["text"]
        if transcript_output_format in ("words_only", "both"):
            current_group["words"] = segments[0]["words"]

        for i in range(1, len(segments)):
            # Calculate time gap between consecutive segments
            time_gap = segments[i]["start"] - segments[i - 1]["end"]

            # If the current segment's speaker is the same as the previous segment's speaker,
            # and the time gap is less than or equal to 2 seconds, group them
            if segments[i]["speaker"] == segments[i - 1]["speaker"] and time_gap <= 2 and group_segments:
                current_group["end"] = segments[i]["end"]
                if transcript_output_format in ("segments_only", "both"):
                    current_group["text"] += " " + segments[i]["text"]
                if transcript_output_format in ("words_only", "both"):
                    current_group.setdefault("words", []).extend(segments[i]["words"])
            else:
                # Add the current_group to the output list
                output.append(current_group)

                # Start a new group with the current segment
                current_group = {
                    "start": segments[i]["start"],
                    "end": segments[i]["end"],
                    "speaker": segments[i]["speaker"],
                    "avg_logprob": segments[i]["avg_logprob"],
                }
                if transcript_output_format in ("segments_only", "both"):
                    current_group["text"] = segments[i]["text"]
                if transcript_output_format in ("words_only", "both"):
                    current_group["words"] = segments[i]["words"]

        # Add the last group to the output list
        output.append(current_group)

        time_cleaning_end = time.time()
        print(
            f"Finished with cleaning, took {time_cleaning_end - time_merging_end:.5} seconds"
        )
        time_end = time.time()
        time_diff = time_end - time_start

        system_info = f"""Processing time: {time_diff:.5} seconds"""
        print(system_info)
        self._update_progress("清理和格式化结果...", 90)
        return output, detected_num_speakers, transcript_info.language