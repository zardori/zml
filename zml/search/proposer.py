"""OpenRouter-backed prompt proposer for the partial-fire search loop.

OpenRouter exposes an OpenAI-compatible API, so any model can be wired in by changing the model id
in the config. Each round the proposer is shown a compact digest of what worked (and what didn't)
and asked for a fresh batch of prompts that should render a single clean fire ignition partway
through a short clip.
"""

import json
import os
import re
from dataclasses import dataclass

from openai import OpenAI

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

SYSTEM_PROMPT = """\
You write text-to-video prompts for CogVideoX. Each prompt produces a ~6 second, 49-frame clip.

Your goal: prompts that render PARTIAL fire — the clip must contain a fire-free part AND a part
with clearly visible flames, separated by ONE clean ignition. Aim for the ignition to land roughly
in the MIDDLE of the clip (first half no fire, second half fire), though earlier/later is still ok.

Hard rules for every prompt:
- Describe a single, static-camera scene with one clear ignition event partway through.
- The first part: explicitly state there is absolutely no fire, no smoke, no sparks, no embers.
- The second part: clearly visible, realistic orange flames.
- Keep it bright/daylight and photorealistic so the video stays high quality and on-prompt.
- Avoid clips that would be on fire the whole time, or never catch fire at all.
- Vary the scene, subject and setting across the batch — do not repeat one template.

Respond with ONLY a JSON array of {k} prompt strings. No prose, no markdown fences."""


@dataclass
class PairFeedback:
    prompt: str
    seed: int
    separation_score: float
    onset_frame: int | None
    fire_fraction: float
    num_transitions: int
    clip_score: float
    accepted: bool


@dataclass
class ProposerFeedback:
    round_index: int
    total_evaluated: int
    total_accepted: int
    top: list[PairFeedback]
    bottom: list[PairFeedback]


@dataclass
class ProposeResult:
    prompts: list[str]
    raw_response: str
    request_digest: str


class PromptProposer:
    def __init__(
        self,
        model: str,
        seed_examples: list[str],
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 1.0,
    ) -> None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set; cannot reach the proposer model.")
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.seed_examples = seed_examples
        self.temperature = temperature

    def propose(self, feedback: ProposerFeedback | None, k: int) -> ProposeResult:
        user_msg = self._build_user_message(feedback, k)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(k=k)},
            {"role": "user", "content": user_msg},
        ]
        raw = self._call(messages)
        prompts = self._parse(raw, k)
        if not prompts:  # one retry nudging the format, then fall back to known-good prompts
            raw = self._call(messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"Return ONLY a JSON array of exactly {k} strings."},
            ])
            prompts = self._parse(raw, k)
        if not prompts:
            prompts = self._fallback(feedback, k)
        return ProposeResult(prompts=prompts[:k], raw_response=raw, request_digest=user_msg)

    def _call(self, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature
        )
        return resp.choices[0].message.content or ""

    def _build_user_message(self, feedback: ProposerFeedback | None, k: int) -> str:
        if feedback is None or feedback.total_evaluated == 0:
            examples = "\n".join(f"- {p}" for p in self.seed_examples[:4])
            return (
                f"Here are example prompts that previously produced partial fire:\n{examples}\n\n"
                f"Propose {k} NEW, diverse prompts following the rules."
            )
        return (
            f"Round {feedback.round_index}. So far {feedback.total_accepted}/"
            f"{feedback.total_evaluated} (prompt, seed) pairs were accepted.\n\n"
            f"BEST results so far (high separation_score = clean partial fire):\n"
            f"{self._format_pairs(feedback.top)}\n\n"
            f"WORST results (learn what to avoid):\n"
            f"{self._format_pairs(feedback.bottom)}\n\n"
            f"separation_score is 0..1 (1 = a single clean mid-clip ignition). onset_frame is the "
            f"first fire frame out of 49 (~24 is the middle). fire_fraction is the share of frames "
            f"with fire (aim ~0.5). num_transitions>1 means flicker (bad).\n\n"
            f"Propose {k} NEW, diverse prompts that should improve on the best results."
        )

    @staticmethod
    def _format_pairs(pairs: list[PairFeedback]) -> str:
        lines = []
        for p in pairs:
            snippet = p.prompt if len(p.prompt) <= 160 else p.prompt[:157] + "..."
            lines.append(
                f"- sep={p.separation_score:.2f} onset={p.onset_frame} "
                f"fire_frac={p.fire_fraction:.2f} transitions={p.num_transitions} "
                f"clip={p.clip_score:.2f} | {snippet}"
            )
        return "\n".join(lines) if lines else "(none)"

    @staticmethod
    def _parse(raw: str, k: int) -> list[str]:
        text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        match = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        return [str(p).strip() for p in data if isinstance(p, str) and p.strip()]

    @staticmethod
    def _fallback(feedback: ProposerFeedback | None, k: int) -> list[str]:
        if feedback and feedback.top:
            return [p.prompt for p in feedback.top][:k]
        return []


def feedback_pair(prompt: str, seed: int, metrics) -> PairFeedback:
    """Build a PairFeedback row from a PartialFireMetrics (avoids importing scorer here)."""
    return PairFeedback(
        prompt=prompt,
        seed=seed,
        separation_score=metrics.separation_score,
        onset_frame=metrics.onset_frame,
        fire_fraction=metrics.fire_fraction,
        num_transitions=metrics.num_transitions,
        clip_score=metrics.clip_score,
        accepted=metrics.accepted,
    )
