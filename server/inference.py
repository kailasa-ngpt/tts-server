import io
import os
from typing import Optional, Tuple

import numpy as np
import soundfile as sf


class TTSBackend:
    def load_model(self) -> None:
        raise NotImplementedError

    def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        voice: Optional[str] = None,
        speaker_wav_bytes: Optional[bytes] = None,
    ) -> Tuple[np.ndarray, int]:
        """Return (audio_float32_mono, sample_rate)."""
        raise NotImplementedError


class UnslothCSMBackend(TTSBackend):
    def __init__(self, model_id: Optional[str] = None, device: Optional[str] = None) -> None:
        # Default to Unsloth model id
        self.model_id = model_id or os.getenv("CSM_MODEL_ID", "unsloth/csm-1b")
        self.device = device
        self.model = None
        self.processor = None

    def load_model(self) -> None:
        # Import unsloth FIRST to ensure it patches dependencies
        try:
            from unsloth import FastModel  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Unsloth CSM backend requires 'unsloth' and 'unsloth_zoo'.\n"
                "Install with: pip install unsloth unsloth_zoo and install torch from https://pytorch.org/get-started/locally/"
            ) from e

        try:
            from transformers import CsmForConditionalGeneration  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Unsloth CSM backend requires 'transformers'. Install with: pip install transformers"
            ) from e

        if self.device is None:
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"

        # Load model + processor via Unsloth FastModel
        from unsloth import FastModel  # type: ignore  # re-import for static checkers
        self.model, self.processor = FastModel.from_pretrained(
            model_name=self.model_id,
            max_seq_length=int(os.getenv("CSM_MAX_SEQ_LEN", "2048")),
            dtype=None,
            auto_model=CsmForConditionalGeneration,
            load_in_4bit=os.getenv("CSM_LOAD_IN_4BIT", "false").lower() == "true",
        )

        try:
            self.model.to(self.device)
        except Exception:
            pass
        try:
            self.model.eval()
        except Exception:
            pass

    def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        voice: Optional[str] = None,
        speaker_wav_bytes: Optional[bytes] = None,
    ) -> Tuple[np.ndarray, int]:
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Optional reference audio
        speaker_audio = None
        speaker_sr = None
        if speaker_wav_bytes is not None:
            import io as _io

            with sf.SoundFile(_io.BytesIO(speaker_wav_bytes)) as f:
                speaker_audio = f.read(dtype="float32")
                speaker_sr = f.samplerate
            if speaker_audio.ndim > 1:
                speaker_audio = np.mean(speaker_audio, axis=1)

        # Build inputs following Transformers CSM docs: simple text prompt with speaker id, no audio tokens
        # Default speaker id "0" unless provided via `voice`
        try:
            speaker_id = str(voice) if voice is not None else "0"
        except Exception:
            speaker_id = "0"
        # Strip any stray audio special tokens from user text
        sanitized_text = (
            text.replace("<|AUDIO|>", "").replace("<|audio_eos|>", "")
        )
        prompt = f"[{speaker_id}]{sanitized_text}"
        # Use processor(text, add_special_tokens=True) as per docs
        inputs = self.processor(prompt, add_special_tokens=True, return_tensors="pt")
        # If no speaker audio, strip any accidental audio special tokens from tokenized inputs
        if speaker_audio is None:
            try:
                import torch

                audio_id = getattr(self.processor, "audio_token_id", None)
                audio_eos_id = getattr(self.processor, "audio_eos_token_id", None)
                if audio_id is None:
                    audio_id = self.processor.tokenizer.convert_tokens_to_ids("<|AUDIO|>")
                if audio_eos_id is None:
                    audio_eos_id = self.processor.tokenizer.convert_tokens_to_ids("<|audio_eos|>")
                ids = inputs.get("input_ids", None)
                if ids is not None:
                    mask = torch.ones_like(ids, dtype=torch.bool)
                    if audio_id is not None and audio_id != -1:
                        mask &= ids != audio_id
                    if audio_eos_id is not None and audio_eos_id != -1:
                        mask &= ids != audio_eos_id
                    filtered = ids[mask]
                    inputs["input_ids"] = filtered.view(1, -1)
                    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"]) if "attention_mask" in inputs else torch.ones_like(inputs["input_ids"]) 
            except Exception:
                pass

        # Move inputs to device
        try:
            import torch

            inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        except Exception:
            pass

        # Generate audio
        max_new = int(os.getenv("CSM_MAX_NEW_TOKENS", "1024"))
        outputs = self.model.generate(
            **{k: v for k, v in inputs.items() if k in ("input_ids", "input_values", "input_values_cutoffs")},
            max_new_tokens=max_new,
            output_audio=True,
        )

        # Extract audio and sample rate
        audio_tensor = None
        sr = 24000
        if hasattr(outputs, "audio") and outputs.audio is not None:
            audio_list = outputs.audio
            if isinstance(audio_list, (list, tuple)) and len(audio_list) > 0:
                audio_tensor = audio_list[0]
                try:
                    sr = int(getattr(self.processor, "sampling_rate", 24000) or 24000)
                except Exception:
                    sr = 24000

        if audio_tensor is None:
            raise RuntimeError("Unsloth CSM: No audio returned from generation. Verify prompt and model id.")

        try:
            audio = audio_tensor.detach().cpu().numpy().astype(np.float32).squeeze()
        except Exception:
            audio = np.array(audio_tensor, dtype=np.float32).squeeze()

        return audio, sr


class UnslothCSMBackend(TTSBackend):
    def __init__(self, model_id: Optional[str] = None, device: Optional[str] = None) -> None:
        # Default to Unsloth model id
        self.model_id = model_id or os.getenv("CSM_MODEL_ID", "unsloth/csm-1b")
        self.device = device
        self.model = None
        self.processor = None

    def load_model(self) -> None:
        # Import unsloth FIRST to ensure it patches dependencies
        try:
            from unsloth import FastModel  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Unsloth CSM backend requires 'unsloth' and 'unsloth_zoo'.\n"
                "Install with: pip install unsloth unsloth_zoo and install torch from https://pytorch.org/get-started/locally/"
            ) from e

        try:
            from transformers import CsmForConditionalGeneration  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Unsloth CSM backend requires 'transformers'. Install with: pip install transformers"
            ) from e

        if self.device is None:
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"

        # Load model + processor via Unsloth FastModel
        self.model, self.processor = FastModel.from_pretrained(
            model_name=self.model_id,
            max_seq_length=int(os.getenv("CSM_MAX_SEQ_LEN", "2048")),
            dtype=None,
            auto_model=CsmForConditionalGeneration,
            load_in_4bit=os.getenv("CSM_LOAD_IN_4BIT", "false").lower() == "true",
        )

        try:
            self.model.to(self.device)
        except Exception:
            pass
        try:
            self.model.eval()
        except Exception:
            pass

    def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        voice: Optional[str] = None,
        speaker_wav_bytes: Optional[bytes] = None,
    ) -> Tuple[np.ndarray, int]:
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Optional reference audio
        speaker_audio = None
        speaker_sr = None
        if speaker_wav_bytes is not None:
            import io as _io

            with sf.SoundFile(_io.BytesIO(speaker_wav_bytes)) as f:
                speaker_audio = f.read(dtype="float32")
                speaker_sr = f.samplerate
            if speaker_audio.ndim > 1:
                speaker_audio = np.mean(speaker_audio, axis=1)

        # Compose prompt for CSM: BOS + text + EOS + AUDIO token
        try:
            speaker_id = str(voice) if voice is not None else "0"
        except Exception:
            speaker_id = "0"
        sanitized_text = (
            text.replace("<|AUDIO|>", "").replace("<|audio_eos|>", "")
        )
        prompt = f"[{speaker_id}]{sanitized_text}"
        inputs = self.processor(prompt, add_special_tokens=True, return_tensors="pt")
        if speaker_audio is None:
            try:
                import torch

                audio_id = getattr(self.processor, "audio_token_id", None)
                audio_eos_id = getattr(self.processor, "audio_eos_token_id", None)
                if audio_id is None:
                    audio_id = self.processor.tokenizer.convert_tokens_to_ids("<|AUDIO|>")
                if audio_eos_id is None:
                    audio_eos_id = self.processor.tokenizer.convert_tokens_to_ids("<|audio_eos|>")
                ids = inputs.get("input_ids", None)
                if ids is not None:
                    mask = torch.ones_like(ids, dtype=torch.bool)
                    if audio_id is not None and audio_id != -1:
                        mask &= ids != audio_id
                    if audio_eos_id is not None and audio_eos_id != -1:
                        mask &= ids != audio_eos_id
                    filtered = ids[mask]
                    inputs["input_ids"] = filtered.view(1, -1)
                    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"]) if "attention_mask" in inputs else torch.ones_like(inputs["input_ids"]) 
            except Exception:
                pass

        # Move inputs to device
        try:
            import torch

            inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        except Exception:
            pass

        # Generate audio
        max_new = int(os.getenv("CSM_MAX_NEW_TOKENS", "1024"))
        outputs = self.model.generate(
            **{k: v for k, v in inputs.items() if k in ("input_ids", "input_values", "input_values_cutoffs")},
            max_new_tokens=max_new,
            output_audio=True,
        )

        # Extract audio and sample rate
        audio_tensor = None
        sr = 24000
        if hasattr(outputs, "audio") and outputs.audio is not None:
            audio_list = outputs.audio
            if isinstance(audio_list, (list, tuple)) and len(audio_list) > 0:
                audio_tensor = audio_list[0]
                try:
                    sr = int(getattr(self.processor, "sampling_rate", 24000) or 24000)
                except Exception:
                    sr = 24000

        if audio_tensor is None:
            raise RuntimeError("Unsloth CSM: No audio returned from generation. Verify prompt and model id.")

        try:
            audio = audio_tensor.detach().cpu().numpy().astype(np.float32).squeeze()
        except Exception:
            audio = np.array(audio_tensor, dtype=np.float32).squeeze()

        return audio, sr


class TransformersCSMBackend(TTSBackend):
    def __init__(self, model_id: Optional[str] = None, device: Optional[str] = None) -> None:
        self.model_id = model_id or os.getenv("CSM_MODEL_ID", "sesame/csm-1b")
        self.device = device
        self.model = None
        self.processor = None

    def load_model(self) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoProcessor, CsmForConditionalGeneration  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Transformers CSM backend requires 'torch' and 'transformers' packages. "
                "Install with: pip install transformers and install torch from https://pytorch.org/get-started/locally/"
            ) from e

        if self.device is None:
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"

        from transformers import AutoProcessor, CsmForConditionalGeneration  # type: ignore

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = CsmForConditionalGeneration.from_pretrained(self.model_id)

        try:
            self.model.to(self.device)
        except Exception:
            pass
        try:
            self.model.eval()
        except Exception:
            pass

    def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        voice: Optional[str] = None,
        speaker_wav_bytes: Optional[bytes] = None,
    ) -> Tuple[np.ndarray, int]:
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        speaker_audio = None
        speaker_sr = None
        if speaker_wav_bytes is not None:
            import io as _io

            with sf.SoundFile(_io.BytesIO(speaker_wav_bytes)) as f:
                speaker_audio = f.read(dtype="float32")
                speaker_sr = f.samplerate
            if speaker_audio.ndim > 1:
                speaker_audio = np.mean(speaker_audio, axis=1)

        prompt = text
        try:
            bot = getattr(self.processor.tokenizer, "bos_token", None) or "<|begin_of_text|>"
            eot = getattr(self.processor.tokenizer, "eos_token", None) or "<|end_of_text|>"
            audio_token = getattr(self.processor, "audio_token", "<|AUDIO|>")
            prompt = f"{bot}{text}{eot}{audio_token}"
        except Exception:
            pass

        proc_kwargs = {
            "text": [prompt],
            "common_kwargs": {"return_tensors": "pt"},
        }
        if speaker_audio is not None and speaker_sr is not None:
            proc_kwargs["audio"] = speaker_audio
            proc_kwargs["audio_kwargs"] = {"sampling_rate": int(speaker_sr)}

        inputs = self.processor(**proc_kwargs)

        try:
            import torch

            inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        except Exception:
            pass

        outputs = self.model.generate(
            **{k: v for k, v in inputs.items() if k in ("input_ids", "input_values", "input_values_cutoffs")},
            max_new_tokens=int(os.getenv("CSM_MAX_NEW_TOKENS", "1024")),
            output_audio=True,
        )

        audio_tensor = None
        sr = 24000
        if hasattr(outputs, "audio") and outputs.audio is not None:
            audio_list = outputs.audio
            if isinstance(audio_list, (list, tuple)) and len(audio_list) > 0:
                audio_tensor = audio_list[0]
                try:
                    sr = int(getattr(self.processor, "sampling_rate", 24000) or 24000)
                except Exception:
                    sr = 24000

        if audio_tensor is None:
            raise RuntimeError("Transformers CSM: No audio returned from generation. Verify prompt and model id.")

        try:
            audio = audio_tensor.detach().cpu().numpy().astype(np.float32).squeeze()
        except Exception:
            audio = np.array(audio_tensor, dtype=np.float32).squeeze()

        return audio, sr


def get_backend() -> TTSBackend:
    # Default to pure Transformers CSM for CPU-first setups
    backend_name = os.getenv("TTS_BACKEND", "transformers_csm").lower()
    device = os.getenv("DEVICE", None)

    if backend_name in ("transformers_csm", "transformers", "sesame_csm", "csm", "sesame"):
        return TransformersCSMBackend(device=device)
    if backend_name in ("unsloth_csm", "unsloth"):
        return UnslothCSMBackend(device=device)

    raise ValueError(f"Unsupported TTS_BACKEND: {backend_name}")


def load_model() -> TTSBackend:
    backend = get_backend()
    backend.load_model()
    return backend


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    # Clamp/convert to float32 mono
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()
