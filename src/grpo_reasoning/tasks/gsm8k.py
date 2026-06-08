from __future__ import annotations

from dataclasses import dataclass, field

from datasets import load_dataset

from .base import Task, register_task


@dataclass
class GSM8KTask(Task):
    """GSM8K arithmetic reasoning task.

    Args:
        name: Registered task name.
        task_instructions: Additional system prompt instructions.
        few_shot_question: Example user question.
        few_shot_answer: Example assistant answer.
        _subset: Hugging Face GSM8K subset name.
    """

    name: str = "gsm8k"
    task_instructions: str = "The answer must be a single integer."
    few_shot_question: str = "What is 2+2?"
    few_shot_answer: str = (
        "<reasoning>To calculate 2+2, we simply add the numbers together: "
        "2 + 2 = 4.</reasoning>\n<answer>4</answer>"
    )

    _subset: str = field(default="main", repr=False)

    def load_raw(self, split: str):
        """Load the GSM8K split from Hugging Face.

        Args:
            split: Dataset split name.

        Returns:
            Hugging Face dataset for the requested split.
        """
        return load_dataset("openai/gsm8k", self._subset, split=split)

    def extract_question(self, row: dict) -> str:
        """Extract a GSM8K question.

        Args:
            row: Raw GSM8K row.

        Returns:
            Question text.
        """
        return row["question"]

    def extract_answer(self, row: dict) -> str:
        """Extract the final GSM8K answer.

        Args:
            row: Raw GSM8K row.

        Returns:
            Final answer after the `####` delimiter, or an empty string if absent.
        """
        gold = row["answer"]
        try:
            return gold.split("####")[1].strip()
        except IndexError:
            return ""


register_task("gsm8k")(GSM8KTask)
