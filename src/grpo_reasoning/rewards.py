from __future__ import annotations

import re
from typing import Callable

from .prompts import extract_xml_answer

# Strict R1-style format: exactly one <reasoning>...</reasoning>\n<answer>...</answer>,
_FORMAT_PATTERN = re.compile(
    r"^<reasoning>(?:(?!</reasoning>).)*</reasoning>\n<answer>(?:(?!</answer>).)*</answer>$",
    re.DOTALL,
)


def _completion_text(completion) -> str:
    """Normalize a completion object to text.

    Args:
        completion: TRL completion object or plain string.

    Returns:
        Assistant content text when available, otherwise `str(completion)`.
    """
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get("content", "")
    return str(completion)


def format_reward(completions, **_) -> list[float]:
    """Score strict R1-style output format.

    Args:
        completions: Batch of completion objects from TRL.
        **_: Extra reward-function keyword arguments ignored by this reward.

    Returns:
        Reward list with 1.0 for strict format matches and 0.0 otherwise.
    """
    texts = [_completion_text(c) for c in completions]
    return [1.0 if _FORMAT_PATTERN.match(t) else 0.0 for t in texts]


def soft_format_reward(completions, **_) -> list[float]:
    """Score loose presence of reasoning and answer tags.

    Args:
        completions: Batch of completion objects from TRL.
        **_: Extra reward-function keyword arguments ignored by this reward.

    Returns:
        Reward list with 0.5 when both tag pairs appear and 0.0 otherwise.
    """
    texts = [_completion_text(c) for c in completions]
    out = []
    for t in texts:
        has_reasoning = "<reasoning>" in t and "</reasoning>" in t
        has_answer = "<answer>" in t and "</answer>" in t
        out.append(0.5 if (has_reasoning and has_answer) else 0.0)
    return out


def make_exact_match_reward(weight: float = 2.0) -> Callable:
    """Create an exact-match correctness reward.

    Args:
        weight: Reward value assigned to each exact answer match.

    Returns:
        Reward function that compares extracted answers with gold answers.
    """

    def correctness_reward(completions, answer, **_) -> list[float]:
        """Score extracted answers by exact string match.

        Args:
            completions: Batch of completion objects from TRL.
            answer: Batch of gold answer strings.
            **_: Extra reward-function keyword arguments ignored by this reward.

        Returns:
            Reward list with `weight` for matches and 0.0 otherwise.
        """
        texts = [_completion_text(c) for c in completions]
        extracted = [extract_xml_answer(t) for t in texts]
        return [weight if e == a else 0.0 for e, a in zip(extracted, answer)]

    correctness_reward.__name__ = "correctness_reward"
    return correctness_reward


def make_moleculariq_reward(task_type: str, weight: float = 2.0) -> Callable:
    """Create a MolecularIQ correctness reward.

    Args:
        task_type: MolecularIQ task variant used for scoring.
        weight: Reward value assigned when MolecularIQ scoring returns a full match.

    Returns:
        Reward function that scores completions via `moleculariq_core.evaluate_answer`.
    """
    import json

    from moleculariq_core import evaluate_answer

    def correctness_reward(completions, answer, **_) -> list[float]:
        """Score MolecularIQ answers against JSON-encoded targets.

        Args:
            completions: Batch of completion objects from TRL.
            answer: Batch of JSON-encoded MolecularIQ targets.
            **_: Extra reward-function keyword arguments ignored by this reward.

        Returns:
            Reward list with `weight` for full MolecularIQ matches and 0.0 otherwise.
        """
        texts = [_completion_text(c) for c in completions]
        scores: list[float] = []
        for text, gold_json in zip(texts, answer):
            extracted = extract_xml_answer(text)
            if not extracted:
                scores.append(0.0)
                continue
            try:
                target = (
                    json.loads(gold_json) if isinstance(gold_json, str) else gold_json
                )
            except json.JSONDecodeError:
                scores.append(0.0)
                continue
            try:
                if task_type == "constraint_generation":
                    s = float(
                        evaluate_answer(
                            task_type=task_type,
                            predicted=extracted,
                            constraints=target,
                        )
                    )
                else:
                    s = float(
                        evaluate_answer(
                            task_type=task_type, predicted=extracted, target=target
                        )
                    )
            except Exception:
                s = 0.0
            scores.append(weight if s >= 1.0 else 0.0)
        return scores

    correctness_reward.__name__ = "moleculariq_correctness_reward"
    return correctness_reward
