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


def test_render_prompt_26_choices_ends_with_z():
    text = render_prompt(q(choices=[f"opt{i}" for i in range(26)], answer=0).validate())
    lines = text.split("\n")
    assert lines[-3].startswith("Z) ")
    assert lines[-2] == "Answer with the letter only."


def test_validate_rejects_unknown_primitive():
    with pytest.raises(ValueError, match="primitive"):
        q(primitive="rank").validate()


def test_validate_rejects_bad_cardinality():
    with pytest.raises(ValueError, match="choices"):
        q(choices=["only"], answer=0).validate()
    with pytest.raises(ValueError, match="choices"):
        q(choices=[str(i) for i in range(27)], answer=0).validate()


def test_validate_rejects_non_list_choices():
    with pytest.raises(ValueError, match="list of str"):
        q(choices="abc").validate()
    with pytest.raises(ValueError, match="list of str"):
        q(choices=["a", 1, "b"]).validate()


def test_validate_rejects_empty_choice():
    with pytest.raises(ValueError, match="non-empty single-line"):
        q(choices=["a", "", "b"]).validate()
    with pytest.raises(ValueError, match="non-empty single-line"):
        q(choices=["a", "   ", "b"]).validate()


def test_validate_rejects_multiline_choice():
    with pytest.raises(ValueError, match="non-empty single-line"):
        q(choices=["a", "b\nc", "d"]).validate()
    with pytest.raises(ValueError, match="non-empty single-line"):
        q(choices=["a", "b\rc", "d"]).validate()


def test_validate_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        q(choices=["a", "a", "b"]).validate()


def test_validate_rejects_duplicates_after_strip():
    with pytest.raises(ValueError, match="duplicate"):
        q(choices=["a", "a ", "b"]).validate()


def test_validate_noul_choices():
    q(primitive="noul", choices=["true", "false"], answer=0).validate()
    with pytest.raises(ValueError, match="noul"):
        q(primitive="noul", choices=["yes", "no"], answer=0).validate()
    with pytest.raises(ValueError, match="noul"):
        q(primitive="noul", choices=["false", "true"], answer=0).validate()


def test_validate_score_needs_ordered():
    with pytest.raises(ValueError, match="ordered"):
        q(primitive="score", ordered=False).validate()
    q(primitive="score", ordered=True).validate()


def test_validate_answer_range():
    with pytest.raises(ValueError, match="answer"):
        q(answer=3).validate()
    with pytest.raises(ValueError, match="answer"):
        q(answer=-1).validate()


def test_validate_answer_must_be_int():
    with pytest.raises(ValueError, match="answer"):
        q(answer=True).validate()
    with pytest.raises(ValueError, match="answer"):
        q(answer=1.0).validate()


def test_validate_answer_none_ok():
    assert q(answer=None).validate().answer is None


def test_json_roundtrip():
    original = q(source="unit", id="x1")
    assert Question.from_json(original.to_json()) == original


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / "qs.jsonl"
    items = [q(id="1"), q(id="2", choices=["x", "y"], answer=0)]
    write_jsonl(path, items)
    assert read_jsonl(path) == items


def test_jsonl_roundtrip_unicode_and_crlf(tmp_path):
    path = tmp_path / "qs.jsonl"
    ctx = "line one\r\nline two caf\u00e9 \U0001F600"
    items = [q(id="u", context=ctx)]
    write_jsonl(path, items)
    back = read_jsonl(path)
    assert back == items
    assert back[0].context == ctx


def test_read_jsonl_error_includes_line_number(tmp_path):
    path = tmp_path / "qs.jsonl"
    good = q(id="1").to_json()
    bad = q(id="2").to_json().replace('"answer": 1', '"answer": 9')
    assert bad != q(id="2").to_json()
    path.write_text(good + "\n" + bad + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"qs\.jsonl:2: .*answer"):
        read_jsonl(path)
    path.write_text(good + "\n\n{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"qs\.jsonl:3: "):
        read_jsonl(path)


def test_write_jsonl_invalid_row_leaves_no_file(tmp_path):
    path = tmp_path / "qs.jsonl"
    with pytest.raises(ValueError, match="answer"):
        write_jsonl(path, [q(id="1"), q(id="2", answer=7)])
    assert not path.exists()


def test_k():
    assert q().k == 3


def test_letter_token_ids_gpt2():
    tok = pytest.importorskip("transformers").AutoTokenizer.from_pretrained("gpt2")
    ids = letter_token_ids(tok)
    assert ids[:4] == [317, 347, 327, 360]
    assert len(set(ids)) == 26


def test_nota_constant():
    assert NOTA == "None of the above"
