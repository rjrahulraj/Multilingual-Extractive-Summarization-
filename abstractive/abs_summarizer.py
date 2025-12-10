# abs_summarizer.py — Ultra-Speed + Safe Generation Version

from typing import List, Optional
import torch
import threading
import time
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def _get_device() -> torch.device:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ABS] Using device: {dev}")
    return dev


# ------------------------------------------------------------------
# 🔥 Windows-safe Timeout Wrapper (no SIGALRM needed)
# ------------------------------------------------------------------
class TimeoutException(Exception):
    pass


def run_with_timeout(func, timeout_sec, *args, **kwargs):
    """
    Runs a function with a timeout (Windows compatible using threads).
    """
    result_container = {"result": None, "error": None}

    def wrapper():
        try:
            result_container["result"] = func(*args, **kwargs)
        except Exception as e:
            result_container["error"] = e

    thread = threading.Thread(target=wrapper)
    thread.daemon = True
    thread.start()
    thread.join(timeout_sec)

    if thread.is_alive():
        raise TimeoutException("Generation timed out")

    if result_container["error"] is not None:
        raise result_container["error"]

    return result_container["result"]


# ------------------------------------------------------------------
class AbstractiveSummarizer:
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        max_input_tokens: int = 2048,
        ultra_speed: bool = False,
        use_8bit: bool = False,
    ):
        self.model_name = model_name
        self.device = _get_device()
        self.max_input_tokens = max_input_tokens
        self.ultra_speed = ultra_speed
        self.use_8bit = use_8bit

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        model_kwargs = {}

        if self.use_8bit and self.device.type == "cuda":
            try:
                print("[ABS] Trying to load model in 8-bit…")
                model_kwargs.update({
                    "device_map": "auto",
                    "load_in_8bit": True,
                })
            except Exception as e:
                print("[ABS] 8-bit failed:", e)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)

        if not self.use_8bit:
            self.model.to(self.device)

        # Ultra-speed optimizations
        if self.ultra_speed and self.device.type == "cuda":
            try:
                print("[ABS] Using FP16 + torch.compile()")
                if not self.use_8bit:
                    self.model = self.model.half()
                if hasattr(torch, "compile"):
                    self.model = torch.compile(self.model)
            except Exception as e:
                print("[ABS] Ultra-speed failed:", e)

        self.model.eval()
        torch.set_grad_enabled(False)

    # -----------------------------------------------------
    def _prepare_text(self, text: str, prefix: Optional[str] = None) -> str:
        if prefix:
            return prefix + text
        if "t5" in self.model_name.lower():
            return "summarize: " + text
        return text

    # -----------------------------------------------------
    def _generate_core(self, inputs, max_len, min_len):
        """Actual HF generate call (wrapped by timeout)."""

        return self.model.generate(
            **inputs,
            max_new_tokens=max_len,
            min_length=min_len,
            num_beams=1,       # speed
            do_sample=False,
            early_stopping=True,
        )

    # -----------------------------------------------------
    def _safe_generate(self, inputs, max_len, min_len):
        """
        Safe generation wrapper: protects against freezing, OOM, loops.
        """

        # 1. Try with timeout
        try:
            outputs = run_with_timeout(
                self._generate_core,
                timeout_sec=15,   # ⏳ 15s timeout
                inputs=inputs,
                max_len=max_len,
                min_len=min_len,
            )
            return outputs

        except TimeoutException:
            print("⚠️ [ABS] Timeout — skipping sample.")
            return None

        except torch.cuda.OutOfMemoryError:
            print("⚠️ [ABS] GPU OOM — clearing cache and skipping.")
            torch.cuda.empty_cache()
            return None

        except Exception as e:
            print("⚠️ [ABS] Generation error:", e)
            return None

    # -----------------------------------------------------
    @torch.inference_mode()
    def summarize(self, text: str, max_len: int = 120, min_len: int = 20, prefix: Optional[str] = None) -> str:
        if not text.strip():
            return ""

        text = self._prepare_text(text, prefix)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.device)

        output = self._safe_generate(inputs, max_len, min_len)

        if output is None:
            return "Generation Failed"

        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

    # -----------------------------------------------------
    @torch.inference_mode()
    def summarize_batch(
        self,
        texts: List[str],
        max_len: int = 120,
        min_len: int = 20,
        prefix: Optional[str] = None,
        batch_size: int = 4,
    ) -> List[str]:

        outputs_all = []

        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            processed = [self._prepare_text(t, prefix) for t in chunk]

            inputs = self.tokenizer(
                processed,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
            ).to(self.device)

            output_ids = self._safe_generate(inputs, max_len, min_len)

            if output_ids is None:
                outputs_all += ["Generation Failed"] * len(chunk)
                continue

            for ids in output_ids:
                outputs_all.append(
                    self.tokenizer.decode(ids, skip_special_tokens=True).strip()
                )

        return outputs_all
