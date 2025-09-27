from __future__ import annotations
import re
from typing import Literal, Optional, Tuple, List
from pathlib import Path

import torch
from osc_llm import LLM, SamplingParams
import numpy as np
from loguru import logger

from .base import TTSModel, Voice
from ..audio import SparkDeTokenizer, SparkTokenizer


class SparkTTS(TTSModel):
    def __init__(
        self,
        checkpoint_dir: str,
        sample_rate: int = 16000,
        device: str = "cuda",
        gpu_memory_utilization: float = 0.5,
    ) -> None:
        super().__init__(checkpoint_dir, sample_rate, device, gpu_memory_utilization)
        self.audio_tokenizer = SparkTokenizer(
            model_path=checkpoint_dir, device=device, dtype=torch.float16
        )
        self.audio_detokenizer = SparkDeTokenizer(
            model_path=checkpoint_dir,
            device=device,
            dtype=torch.float16,
        )

    def init_llm(
        self,
        checkpoint_dir: str,
        device: str = "cuda",
        gpu_memory_utilization: float = 0.9,
    ) -> LLM:
        checkpoint_dir = Path(checkpoint_dir) / "LLM"
        llm = LLM(
            checkpoint_dir=checkpoint_dir,
            device=device,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        return llm

    def create_voice(
        self,
        name: str,
        gender: Literal["female", "male"],
        text="你好啊，你叫什么名字?",
        pitch: Optional[
            Literal["very_low", "low", "moderate", "high", "very_high"]
        ] = None,
        speed: Optional[
            Literal["very_low", "low", "moderate", "high", "very_high"]
        ] = None,
    ) -> Voice:
        if name in self.voices:
            logger.warning(f"Voice {name} already exists, will be overwritten")
        assert gender in ["female", "male"], "Gender must be female or male"
        voice = Voice(name=name, reference_text=text)
        prompt = self.apply_prompt_control(text, gender, pitch, speed)
        generated_output = self.llm.generate(prompt=prompt)
        pred_semantic_tokens = [
            int(token)
            for token in re.findall(r"bicodec_semantic_(\d+)", generated_output)
        ]
        assert len(pred_semantic_tokens) > 0, "Semantic tokens are not found"
        pred_semantic_ids = torch.tensor(pred_semantic_tokens).to(torch.int32)
        voice.semantic_tokens = pred_semantic_ids
        global_tokens = [
            int(token)
            for token in re.findall(r"bicodec_global_(\d+)", generated_output)
        ]
        assert len(global_tokens) > 0, "Global tokens are not found"
        global_token_ids = torch.tensor(global_tokens).squeeze(0).long()
        voice.global_tokens = global_token_ids
        self.voices[name] = voice
        return voice

    def add_voice(
        self, name: str, audio_path: str, reference_text: Optional[str] = None
    ):
        if name in self.voices:
            logger.warning(f"Voice {name} already exists, will be overwritten")
        if name not in ["female", "male"]:
            tokens = self.audio_tokenizer.tokenize(audio_path)
            voice = Voice(
                name=name,
                global_tokens=tokens["global_tokens"]
                .detach()
                .cpu()
                .squeeze(0)
                .squeeze(0),
                semantic_tokens=tokens["semantic_tokens"].detach().cpu().squeeze(0),
                reference_text=reference_text,
            )
            self.voices[name] = voice
        else:
            logger.error("female or male voice meta data already exists")

    def apply_prompt_control(
        self,
        text: str,
        gender: Literal["female", "male"],
        pitch: Optional[
            Literal["very_low", "low", "moderate", "high", "very_high"]
        ] = None,
        speed: Optional[
            Literal["very_low", "low", "moderate", "high", "very_high"]
        ] = None,
    ):
        assert gender in ["female", "male"], "Gender must be female or male"
        return process_prompt_control(text, gender, pitch, speed)

    def apply_prompt_clone(
        self,
        text: str,
        voice: Voice,
        pitch: Optional[
            Literal["very_low", "low", "moderate", "high", "very_high"]
        ] = None,
        speed: Optional[
            Literal["very_low", "low", "moderate", "high", "very_high"]
        ] = None,
    ):
        return process_prompt(
            text,
            voice.reference_text,
            voice.global_tokens,
            voice.semantic_tokens,
            pitch,
            speed,
        )

    def text2semantic(
        self,
        prompts: List[str],
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        max_generate_tokens: int = 4096,
    ) -> List[str]:
        generated_outputs = self.llm.batch_generate(
            prompts=prompts,
            sampling_params=[
                SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_generate_tokens=max_generate_tokens,
                    repetition_penalty=repetition_penalty,
                )
            ],
        )

        return generated_outputs

    def semantic2wav(
        self,
        semantics: List[torch.Tensor],
        global_tokens: torch.Tensor,
    ) -> List[np.ndarray]:
        """将经过llm生成的语音张量以及全局音色张量转换为音频波形

        Args:
            semantics (List[torch.Tensor]): 经过llm生成的语义表征，一维张量，形状（sequence_length，）
            global_tokens (torch.Tensor): 全局音色，一维张量，形状（global_token_length，）

        Returns:
            List[np.ndarray]: 音频波形
        """
        audios = self.audio_detokenizer.batch_detokenize(
            semantics,
            global_tokens,
        )
        audios = [
            audio.detach().cpu().numpy().astype(np.float32).squeeze(0)
            for audio in audios
        ]
        return audios

    def generate(
        self,
        text: str,
        voice: str | Voice,
        pitch: Optional[
            Literal["very_low", "low", "moderate", "high", "very_high"]
        ] = None,
        speed: Optional[
            Literal["very_low", "low", "moderate", "high", "very_high"]
        ] = None,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1,
        max_generate_tokens: int = 4096,
        length_threshold: int = 50,
        window_size: int = 50,
    ) -> np.ndarray:
        if isinstance(voice, str):
            if voice not in self.voices:
                raise ValueError(f"Voice {voice} not found")
            voice = self.voices[voice]
        else:
            assert isinstance(voice, Voice), "Voice must be a string or Voice object"
        segments = self.preprocess_text(text, length_threshold, window_size)
        prompts = [
            self.apply_prompt_clone(text=segment, voice=voice, pitch=pitch, speed=speed)
            for segment in segments
        ]
        generated_outputs = self.text2semantic(
            prompts, temperature, top_k, top_p, repetition_penalty, max_generate_tokens
        )
        # extract semantic ids
        batch_semantic_ids = []
        for generated_output in generated_outputs:
            semantic_ids = torch.tensor(
                [
                    int(token)
                    for token in re.findall(r"bicodec_semantic_(\d+)", generated_output)
                ]
            ).long()
            if len(semantic_ids) == 0:
                err_msg = "Semantic tokens prediction is empty"
                logger.error(err_msg)
                raise ValueError(err_msg)
            batch_semantic_ids.append(semantic_ids)

        wav = self.semantic2wav(
            batch_semantic_ids,
            global_tokens=voice.global_tokens,
        )
        wav = np.concatenate(wav, axis=0)
        wav = (wav * 32767).astype(np.int16)
        if wav.ndim > 1:
            wav = wav.squeeze()
        return wav


TASK_TOKEN_MAP = {
    "vc": "<|task_vc|>",
    "tts": "<|task_tts|>",
    "asr": "<|task_asr|>",
    "s2s": "<|task_s2s|>",
    "t2s": "<|task_t2s|>",
    "understand": "<|task_understand|>",
    "caption": "<|task_cap|>",
    "controllable_tts": "<|task_controllable_tts|>",
    "prompt_tts": "<|task_prompt_tts|>",
    "speech_edit": "<|task_edit|>",
}

LEVELS_MAP = {
    "very_low": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
    "very_high": 4,
}

GENDER_MAP: dict[Literal["male", "female"], int] = {
    "female": 0,
    "male": 1,
}

ID2GENDER = {v: k for k, v in GENDER_MAP.items()}


def process_prompt(
    text: str,
    prompt_text: Optional[str] = None,
    global_token_ids: torch.Tensor | None = None,
    semantic_token_ids: torch.Tensor = None,
    pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = None,
    speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = None,
) -> Tuple[str, torch.Tensor]:
    """
    Process input for voice cloning.

    Args:
        text: The text input to be converted to speech.
        prompt_text: Transcript of the prompt audio.
        global_token_ids: Global token IDs extracted from reference audio.
        semantic_token_ids: Semantic token IDs extracted from reference audio.
        pitch (str): very_low | low | moderate | high | very_high
        speed (str): very_low | low | moderate | high | very_high

    Returns:
        Tuple containing the formatted input prompt and global token IDs.
    """
    # Convert global tokens to string format
    global_tokens = "".join(
        [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()]
    )

    attribte_tokens = None
    if pitch is not None or speed is not None:
        if pitch is None:
            pitch = "moderate"
        if speed is None:
            speed = "moderate"

        if pitch != "moderate" or speed != "moderate":
            pitch_level_id = LEVELS_MAP[pitch]
            pitch_label_tokens = f"<|pitch_label_{pitch_level_id}|>"
            speed_level_id = LEVELS_MAP[speed]
            speed_label_tokens = f"<|speed_label_{speed_level_id}|>"
            attribte_tokens = "".join([pitch_label_tokens, speed_label_tokens])
    audio_text = (
        text if prompt_text is None or len(prompt_text) == 0 else (prompt_text + text)
    )
    inputs = [TASK_TOKEN_MAP["tts"], "<|start_content|>", audio_text, "<|end_content|>"]
    if attribte_tokens is not None:
        inputs.extend(
            [
                "<|start_style_label|>",
                attribte_tokens,
                "<|end_style_label|>",
            ]
        )
    inputs.extend(
        [
            "<|start_global_token|>",
            global_tokens,
            "<|end_global_token|>",
        ]
    )
    if prompt_text is not None and len(prompt_text) > 0:
        semantic_tokens = "".join(
            [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()]
        )
        inputs.extend(
            [
                "<|start_semantic_token|>",
                semantic_tokens,
            ]
        )
    # Join all input components into a single string
    inputs = "".join(inputs)
    return inputs


def process_prompt_control(
    text: str,
    gender: Optional[Literal["female", "male"]] = "female",
    pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = None,
    speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = None,
):
    """
    Process input for voice creation.

    Args:
        gender (str): female | male.
        pitch (str): very_low | low | moderate | high | very_high
        speed (str): very_low | low | moderate | high | very_high
        text (str): The text input to be converted to speech.

    Return:
        str: Input prompt
    """
    gender = gender or "female"
    pitch = pitch or "moderate"
    speed = speed or "moderate"

    assert gender in GENDER_MAP.keys()
    assert pitch in LEVELS_MAP.keys()
    assert speed in LEVELS_MAP.keys()

    gender_id = GENDER_MAP[gender]
    pitch_level_id = LEVELS_MAP[pitch]
    speed_level_id = LEVELS_MAP[speed]

    pitch_label_tokens = f"<|pitch_label_{pitch_level_id}|>"
    speed_label_tokens = f"<|speed_label_{speed_level_id}|>"
    gender_tokens = f"<|gender_{gender_id}|>"

    attribte_tokens = "".join([gender_tokens, pitch_label_tokens, speed_label_tokens])

    control_tts_inputs = [
        TASK_TOKEN_MAP["controllable_tts"],
        "<|start_content|>",
        text,
        "<|end_content|>",
        "<|start_style_label|>",
        attribte_tokens,
        "<|end_style_label|>",
    ]

    return "".join(control_tts_inputs)
