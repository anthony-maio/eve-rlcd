import pytest

from rlcd.schema import (LETTERS, MAX_CHOICES, NOTA, Question, letter_token_ids,
                         read_jsonl, render_prompt, write_jsonl)


def q(**kw) -> Question:
    base = dict(primitive="choice", context="ctx", question="Which?", choices=["a", "b", "c"], answer=1)
    base.update(kw)
    return Question(**base)


def test_letters():
    assert LETTERS[0] == "A" and LETTERS[-1] == "Z" and len(LETTERS) == MAX_CHOICES == 26


def test_render_prompt_exact():
    text = render_prompt(q(context="  Ticket: printer on fire  ", question="Which department?"))
    assert text == (
        "User: Context:\n"
        "Ticket: printer on fire\n"
        "\n"
        "Question: Which department?\n"
        "Options:\n"
        "A) a\n"
        "B) b\n"
        "C) c\n"
        "Answer with the letter only.\n"
        "Assistant: The answer is"
    )


def test_validate_rejects_bad_cardinality():
    with pytest.raises(ValueError):
        q(choices=["only"]).validate()
    with pytest.raises(ValueError):
        q(choices=[str(i) for i in range(27)], answer=0).validate()


def test_validate_rejects_duplicates():
    with pytest.raises(ValueError):
        q(choices=["a", "a", "b"]).validate()


def test_validate_noul_choices():
    q(primitive="noul", choices=["true", "false"], answer=0).validate()
    with pytest.raises(ValueError):
        q(primitive="noul", choices=["yes", "no"], answer=0).validate()


def test_validate_score_needs_ordered():
    with pytest.raises(ValueError):
        q(primitive="score", ordered=False).validate()
    q(primitive="score", ordered=True).validate()


def test_validate_answer_range():
    with pytest.raises(ValueError):
        q(answer=3).validate()


def test_json_roundtrip():
    original = q(source="unit", id="x1")
    assert Question.from_json(original.to_json()) == original


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / "qs.jsonl"
    items = [q(id="1"), q(id="2", choices=["x", "y"], answer=0)]
    write_jsonl(path, items)
    assert read_jsonl(path) == items


def test_k():
    assert q().k == 3


def test_letter_token_ids_gpt2():
    tok = pytest.importorskip("transformers").AutoTokenizer.from_pretrained("gpt2")
    ids = letter_token_ids(tok)
    assert ids[:4] == [317, 347, 327, 360]
    assert len(set(ids)) == 26


def test_nota_constant():
    assert NOTA == "None of the above"
