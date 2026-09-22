import pytest

from anyjev.question import Question, QuestionError


def test_choice_validation():
    with pytest.raises(QuestionError):
        Question.choice("q", ["only"])
    with pytest.raises(QuestionError):
        Question.choice("q", ["a", "a"])
    with pytest.raises(QuestionError):
        Question.choice("q", [str(i) for i in range(27)])


def test_key_is_stable_and_spec_sensitive():
    a = Question.choice("route?", ["billing", "sales"])
    b = Question.choice("route?", ["billing", "sales"], name="other-name")
    c = Question.choice("route?", ["sales", "billing"])
    assert a.key == b.key          # name is not part of the spec
    assert a.key != c.key          # option order is
    assert a.id == f"q_{a.key[:8]}" and b.id == "other-name"


def test_score_bins():
    q = Question.score("how done?", bins=4, scale=(0, 1))
    assert q.k == 4 and q.ordered
    assert q.bin_centers() == [0.125, 0.375, 0.625, 0.875]
    assert q.options[0] == "0 to 0.25"


def test_noul():
    q = Question.noul("spam?")
    assert q.options == ("Yes", "No")


def test_score_levels():
    q = Question.score("risk?", levels=["benign", "low", "moderate", "high"])
    assert q.k == 4 and q.ordered and q.scale == (0.0, 3.0)
    assert q.bin_centers() == [0.0, 1.0, 2.0, 3.0]
    assert q.key != Question.score("risk?", bins=4).key


def test_score_labels_fall_back_to_letters_when_digits_are_multi_token():
    from anyjev.readout import LabelTokenError, resolve_labels

    class SPTok:  # sentencepiece-like: digits are two pieces, letters and Yes/No are one
        def encode(self, text, add_special_tokens=False):
            t = text.strip()
            if t.isdigit():
                return [7, 100 + int(t)]
            return [200 + (ord(t[0]) if len(t) == 1 else hash(t) % 50)]

    q = Question.score("how urgent?", bins=5)
    labels, ids = resolve_labels(SPTok(), q)
    assert labels == ["A", "B", "C", "D", "E"] and len(set(ids)) == 5
    c = Question.choice("q", ["x", "y"])
    assert resolve_labels(SPTok(), c)[0] == ["A", "B"]

    class Broken:
        def encode(self, text, add_special_tokens=False):
            return [1, 2]

    with pytest.raises(LabelTokenError):
        resolve_labels(Broken(), q)
