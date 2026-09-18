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
        if not isinstance(self.choices, list) or not all(isinstance(c, str) for c in self.choices):
            raise ValueError("choices must be a list of str")
        n = len(self.choices)
        if n < 2 or n > MAX_CHOICES:
            raise ValueError(f"need 2..{MAX_CHOICES} choices, got {n}")
        if any(not c.strip() or "\n" in c or "\r" in c for c in self.choices):
            raise ValueError("choices must be non-empty single-line strings")
        if len({c.strip() for c in self.choices}) != n:
            raise ValueError("duplicate choices")
        if self.primitive == "noul" and self.choices != ["true", "false"]:
            raise ValueError('noul choices must be exactly ["true", "false"]')
        if self.primitive == "score" and not self.ordered:
            raise ValueError("score questions must set ordered=True")
        if self.answer is not None:
            if type(self.answer) is not int:
                raise ValueError(f"answer must be an int or None, got {self.answer!r}")
            if not (0 <= self.answer < n):
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


def render_prefix(context: str) -> str:
    """The part of the prompt that depends on the context only: the header, the stripped
    context, and the blank line before Question. render_prefix(q.context) + render_suffix(q)
    is exactly render_prompt(q)."""
    return f"User: Context:\n{context.strip()}\n\n"


def render_suffix(q: Question) -> str:
    """Everything from Question: onward, so that one context prefix can be shared by many
    questions. See render_prefix."""
    lines = [f"Question: {q.question.strip()}", "Options:"]
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
    out: list[Question] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                out.append(Question.from_json(line))
            except (ValueError, TypeError) as e:
                raise ValueError(f"{path}:{i}: {e}") from e
    return out


def write_jsonl(path, questions: list[Question]) -> None:
    rows = [q.validate().to_json() for q in questions]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(row + "\n")
