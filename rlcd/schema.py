"""Question types, validation, prompt rendering, and letter token ids."""
from __future__ import annotations

import json
import string
from dataclasses import asdict, dataclass
from pathlib import Path

LETTERS: list[str] = list(string.ascii_uppercase)
MAX_CHOICES = 26
NOTA = "None of the above"
NEG = -1e4  # finite mask value for out-of-range letters
PRIMITIVES = ("choice", "score", "noul")


@dataclass
class Question:
    primitive: str
    context: str
    question: str
    choices: list[str]
    ordered: bool = False
    answer: int | None = None
    source: str = ""
    id: str = ""

    @property
    def k(self) -> int:
        return len(self.choices)

    def validate(self) -> "Question":
        if self.primitive not in PRIMITIVES:
            raise ValueError(f"unknown primitive {self.primitive!r}")
        n = len(self.choices)
        if n < 2 or n > MAX_CHOICES:
            raise ValueError(f"need 2..{MAX_CHOICES} choices, got {n}")
        if len(set(self.choices)) != n:
            raise ValueError("duplicate choices")
        if self.primitive == "noul" and self.choices != ["true", "false"]:
            raise ValueError('noul choices must be exactly ["true", "false"]')
        if self.primitive == "score" and not self.ordered:
            raise ValueError("score questions must set ordered=True")
        if self.answer is not None and not (0 <= self.answer < n):
            raise ValueError(f"answer {self.answer} out of range for {n} choices")
        return self

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "Question":
        return cls(**json.loads(s)).validate()


def render_prompt(q: Question) -> str:
    lines = [
        "User: Context:",
        q.context.strip(),
        "",
        f"Question: {q.question.strip()}",
        "Options:",
    ]
    for letter, choice in zip(LETTERS, q.choices):
        lines.append(f"{letter}) {choice}")
    lines.append("Answer with the letter only.")
    lines.append("Assistant: The answer is")
    return "\n".join(lines)


def letter_token_ids(tokenizer) -> list[int]:
    ids: list[int] = []
    for letter in LETTERS:
        toks = tokenizer.encode(" " + letter, add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(f"' {letter}' is not a single token: {toks}")
        ids.append(toks[0])
    if len(set(ids)) != MAX_CHOICES:
        raise ValueError("letter token ids collide")
    return ids


def read_jsonl(path) -> list[Question]:
    with open(path, encoding="utf-8") as f:
        return [Question.from_json(line) for line in f if line.strip()]


def write_jsonl(path, questions: list[Question]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(q.validate().to_json() + "\n")
