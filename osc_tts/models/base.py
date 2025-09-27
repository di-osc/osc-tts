import random
from dataclasses import dataclass
from typing import Literal, Optional, Callable, AsyncIterator, Dict, List

import soundfile as sf
import torch
import numpy as np
from tn.chinese.normalizer import Normalizer as ZhNormalizer
from tn.english.normalizer import Normalizer as EnNormalizer
from osc_llm import LLM

from ..utils import contains_chinese, split_text


@dataclass
class Voice:
    name: str
    global_tokens: torch.Tensor | None = None
    semantic_tokens: torch.Tensor | None = None
    reference_text: str | None = None
    reference_audio: np.ndarray | None = None


class TTSModel:
    """LLM based TTS engine"""

    def __init__(
        self,
        checkpoint_dir: str,
        sample_rate: int = 16000,
        device: str = "cuda",
        gpu_memory_utilization: float = 0.5,
    ) -> None:
        self.llm: LLM = self.init_llm(
            checkpoint_dir=checkpoint_dir,
            device=device,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.sample_rate: int = sample_rate
        self.zh_normalizer: ZhNormalizer = ZhNormalizer(
            overwrite_cache=False, remove_erhua=False, remove_interjections=False
        )
        self.en_normalizer: EnNormalizer = EnNormalizer(overwrite_cache=False)
        self.voices: Dict[str, Voice] = {}
        self.seed: int = 42

    def init_llm(
        self,
        checkpoint_dir: str,
        device: str = "cuda",
        gpu_memory_utilization: float = 0.9,
    ) -> LLM:
        raise NotImplementedError()

    def add_voice(
        self, name: str, audio, reference_text: Optional[str] = None
    ) -> Voice:
        raise NotImplementedError("current model does not support add voice from audio")

    def create_voice(self, gender: Literal["female", "male"]) -> Voice:
        raise NotImplementedError("current model does not support voice creation")

    def generate(
        self,
        text: str,
        voice: Literal["female", "male"] | str = "female",
        pitch: Optional[
            Literal["very_low", "low", "moderate", "high", "very_high"]
        ] = None,
        speed: Optional[
            Literal["very_low", "low", "moderate", "high", "very_high"]
        ] = None,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        max_generate_tokens: int = 4096,
        length_threshold: int = 50,
        window_size: int = 50,
    ) -> np.ndarray:
        raise NotImplementedError()

    def stream(
        self,
        text: str,
        voice: Optional[str] = None,
        pitch: Optional[
            Literal["very_low", "low", "moderate", "high", "very_high"]
        ] = None,
        speed: Optional[
            Literal["very_low", "low", "moderate", "high", "very_high"]
        ] = None,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        max_tokens: int = 4096,
        length_threshold: int = 50,
        window_size: int = 50,
        split_fn: Optional[Callable[[str], list[str]]] = None,
        **kwargs,
    ) -> AsyncIterator[np.ndarray]:
        yield NotImplementedError()

    def list_voices(self) -> list[str]:
        names = []
        for name in self.voices:
            if name not in names:
                names.append(name)
        return names

    def delete_voice(self, name: str):
        if name in self.voices:
            del self.voices[name]

    def get_voice(self, name: str):
        if name in self.voices:
            return self.voices[name]
        else:
            raise ValueError(f"Voice {name} not found")

    def save_voices(self, save_path: str):
        save_data = {
            "class": self.__class__.__name__,
            "voices": self.voices,
        }
        torch.save(save_data, save_path)

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def write_audio(self, audio: np.ndarray, filepath: str):
        sf.write(filepath, audio, self.sample_rate, "PCM_16")

    def load_audio(self, audio_path: str) -> np.ndarray:
        return sf.read(audio_path)[0]

    def preprocess_text(
        self,
        text: str,
        length_threshold: int = 50,
        window_size: int = 50,
        split_fn: Optional[Callable[[str], List[str]]] = None,
    ) -> List[str]:
        """Normalize text and split text into segments

        Args:
            text (str): The text to preprocess
            length_threshold (int, optional): The length threshold of the text. Defaults to 50.
            window_size (int, optional): The window size of the text. Defaults to 50.
            split_fn (Optional[Callable[[str], List[str]]], optional): The function to split the text. Defaults to None.

        Returns:
            List[str]: The preprocessed text
        """
        if contains_chinese(text):
            text = self.zh_normalizer.normalize(text)
        else:
            text = self.en_normalizer.normalize(text)

        tokenize_fn = self.llm.tokenizer.encode
        return split_text(
            text,
            window_size,
            tokenize_fn=tokenize_fn,
            split_fn=split_fn,
            length_threshold=length_threshold,
        )
