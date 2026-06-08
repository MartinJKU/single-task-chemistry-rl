from __future__ import annotations

from grpo_reasoning.rewards import (
    format_reward,
    make_exact_match_reward,
    soft_format_reward,
)


def _conv(text: str):
    """Wrap a string completion as a conversational completion.

    Args:
        text: Assistant completion text.

    Returns:
        Completion object shaped like TRL conversational output.
    """
    return [{"role": "assistant", "content": text}]


def test_format_reward_strict_match():
    """Verify strict format reward accepts only exact scaffold matches.

    Args:
        None.

    Returns:
        None.
    """
    good = _conv("<reasoning>2+2=4</reasoning>\n<answer>4</answer>")
    bad = _conv("The answer is 4")
    assert format_reward([good, bad]) == [1.0, 0.0]


def test_format_reward_rejects_extra_text():
    """Verify strict format reward rejects trailing text.

    Args:
        None.

    Returns:
        None.
    """
    # Trailing text after </answer> should fail strict match
    msg = _conv("<reasoning>x</reasoning>\n<answer>4</answer> extra")
    assert format_reward([msg]) == [0.0]


def test_soft_format_partial_credit():
    """Verify soft format reward gives partial credit for both tag pairs.

    Args:
        None.

    Returns:
        None.
    """
    msg = _conv("<reasoning>x</reasoning> bla <answer>4</answer>")
    assert soft_format_reward([msg]) == [0.5]


def test_correctness_reward_exact_match():
    """Verify exact-match reward scores matching extracted answers.

    Args:
        None.

    Returns:
        None.
    """
    reward = make_exact_match_reward(weight=2.0)
    completions = [
        _conv("<reasoning>...</reasoning>\n<answer>4</answer>"),
        _conv("<reasoning>...</reasoning>\n<answer>7</answer>"),
    ]
    out = reward(completions=completions, answer=["4", "4"])
    assert out == [2.0, 0.0]


def test_correctness_reward_empty_extraction():
    """Verify exact-match reward handles missing answer tags.

    Args:
        None.

    Returns:
        None.
    """
    reward = make_exact_match_reward()
    completions = [_conv("no tags here")]
    assert reward(completions=completions, answer=["42"]) == [0.0]
