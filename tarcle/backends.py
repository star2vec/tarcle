"""Model backends for the behavioral pilot.

The pilot only needs forced-choice scoring: total logprob of each candidate
continuation given a prompt. `LocalHFBackend` is the real path (gpt2 on CPU for
smoke tests, Llama-3.2-3B bf16 on CUDA for the gate). `OpenAICompatBackend` is
an intentionally unimplemented fallback for hosted endpoints.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@dataclass
class ChoiceScores:
    choice_logprobs: list[float]  # total logprob per candidate, in input order
    top_tokens: list[str]  # top-5 next tokens (diagnostic: mass outside candidates?)
    top_logprobs: list[float]


class Backend(Protocol):
    def score_choices(
        self, prompts: list[str], choices_per_prompt: list[list[str]]
    ) -> list[ChoiceScores]: ...


class LocalHFBackend:
    def __init__(
        self, model_name: str, device: str = "cpu", dtype: str = "float32",
        batch_size: int = 8,
    ):
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Right padding: we gather next-token logits at the last real position,
        # which also keeps absolute position ids correct for gpt2.
        self.tokenizer.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=DTYPES[dtype]
        )
        self.model.to(device)
        self.model.eval()
        self._choice_ids: dict[str, list[int]] = {}

    def _encode_choice(self, choice: str) -> list[int]:
        # Candidates are scored as continuations of "...A:", hence the leading space.
        if choice not in self._choice_ids:
            self._choice_ids[choice] = self.tokenizer.encode(
                " " + choice, add_special_tokens=False
            )
        return self._choice_ids[choice]

    @torch.no_grad()
    def _score_continuation(self, prompt: str, cont_ids: list[int]) -> float:
        """Teacher-forced total logprob of a multi-token candidate."""
        prompt_ids = self.tokenizer(prompt)["input_ids"]
        input_ids = torch.tensor([prompt_ids + cont_ids], device=self.device)
        logprobs = torch.log_softmax(self.model(input_ids).logits[0].float(), dim=-1)
        start = len(prompt_ids) - 1
        return sum(
            logprobs[start + j, tok].item() for j, tok in enumerate(cont_ids)
        )

    @torch.no_grad()
    def score_choices(
        self, prompts: list[str], choices_per_prompt: list[list[str]]
    ) -> list[ChoiceScores]:
        results: list[ChoiceScores] = []
        for start in range(0, len(prompts), self.batch_size):
            batch = prompts[start : start + self.batch_size]
            batch_choices = choices_per_prompt[start : start + self.batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True).to(
                self.device
            )
            logits = self.model(**enc).logits
            last = enc["attention_mask"].sum(dim=1) - 1
            next_lp = torch.log_softmax(
                logits[torch.arange(len(batch)), last].float(), dim=-1
            )
            for b, choices in enumerate(batch_choices):
                lp = next_lp[b]
                top_lp, top_ids = lp.topk(5)
                choice_lps = []
                for choice in choices:
                    ids = self._encode_choice(choice)
                    if len(ids) == 1:
                        choice_lps.append(lp[ids[0]].item())
                    else:
                        choice_lps.append(self._score_continuation(batch[b], ids))
                results.append(
                    ChoiceScores(
                        choice_logprobs=choice_lps,
                        top_tokens=[self.tokenizer.decode([i]) for i in top_ids],
                        top_logprobs=[x.item() for x in top_lp],
                    )
                )
        return results


class OpenAICompatBackend:
    """Placeholder for a hosted OpenAI-compatible /v1/completions backend.

    Not used by the local-only pilot. Forced-choice scoring over an API needs
    echo/prompt-logprobs support; implement (with tests) if a provider is chosen.
    """

    def __init__(self, base_url: str, model_name: str, api_key: str = ""):
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key

    def score_choices(
        self, prompts: list[str], choices_per_prompt: list[list[str]]
    ) -> list[ChoiceScores]:
        raise NotImplementedError(
            "Hosted scoring is not implemented: it requires a provider with "
            "echo+logprobs on /v1/completions. The pilot runs on LocalHFBackend."
        )


def build_backend(config) -> Backend:
    if config.backend == "local_hf":
        return LocalHFBackend(
            model_name=config.model,
            device=config.device,
            dtype=config.dtype,
            batch_size=config.batch_size,
        )
    if config.backend == "openai_compat":
        return OpenAICompatBackend(
            base_url=config.base_url, model_name=config.model
        )
    raise ValueError(f"unknown backend: {config.backend}")
