"""UI tests using Streamlit's built-in AppTest. Run with: python -m pytest

Most tests swap in a tiny fake model so the probabilities are known exactly.
The "Real model (V4)" tests use the actual scam_xray_v4.joblib file.
"""

import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

APP_FILE = str(Path(__file__).parent.parent / "app.py")
V4_FILE = Path(APP_FILE).parent / "scam_xray_v4.joblib"
REAL_LOAD = joblib.load  # keep the real function before any test patches it


class FakeModel:
    """Mimics the parts of a scikit-learn classifier that the app uses."""

    def __init__(self, p_ham, classes=("ham", "suspicious")):
        self.classes_ = np.array(classes)
        self._p = {"ham": p_ham, "suspicious": 1 - p_ham}

    def predict_proba(self, texts):
        return np.array([[self._p[c] for c in self.classes_] for _ in texts])


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    """Clear Streamlit's model cache and restore joblib.load around every test."""
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def start_app(monkeypatch, model=None):
    """Start the app. With a model, joblib.load returns it; otherwise the real file is used."""
    if model is not None:
        monkeypatch.setattr(joblib, "load", lambda path: model)
    return AppTest.from_file(APP_FILE, default_timeout=120).run()


def analyze(at, text):
    """Type a message and click Analyze exactly once."""
    at.text_area[0].set_value(text)
    at.button[0].click().run()
    return at


def card(at):
    """Return (css_class, title) of the result card, or None if no card is shown."""
    for md in at.markdown:
        if 'class="result-card' in md.value:
            css = md.value.split('result-card ')[1].split('"')[0]
            title = md.value.split("<h3>")[1].split("</h3>")[0]
            return css, title
    return None


def bars(at):
    return [p.proto.text for p in at.get("progress")]


# ---------- Ham / Suspicious / Uncertain ----------
@pytest.mark.parametrize(
    "p_ham, expected",
    [
        (0.90, "ham"),
        (0.10, "suspicious"),
        (0.60, "ham"),          # difference is exactly 20 points -> definite
        (0.40, "suspicious"),   # exactly 20 points the other way
        (0.5999, "uncertain"),  # just under 20 points
        (0.4001, "uncertain"),
        (0.6001, "ham"),        # just over 20 points
        (0.3999, "suspicious"),
        (0.50, "uncertain"),
        (0.55, "uncertain"),
    ],
)
def test_outcome_and_threshold_boundary(monkeypatch, p_ham, expected):
    at = analyze(start_app(monkeypatch, FakeModel(p_ham)), "some message")
    assert not at.exception
    assert card(at)[0] == expected


def test_uncertain_shows_neutral_warning_and_both_bars(monkeypatch):
    at = analyze(start_app(monkeypatch, FakeModel(0.55)), "some message")
    assert "Uncertain" in card(at)[1]
    assert any("official channel" in w.value for w in at.warning)
    assert bars(at) == ["Suspicious: 45.0%", "Ham: 55.0%"]


def test_definite_result_has_no_uncertain_warning(monkeypatch):
    at = analyze(start_app(monkeypatch, FakeModel(0.90)), "some message")
    assert not at.warning
    assert bars(at) == ["Suspicious: 10.0%", "Ham: 90.0%"]


def test_probabilities_follow_class_names_not_column_order(monkeypatch):
    model = FakeModel(0.90, classes=("suspicious", "ham"))  # reversed column order
    at = analyze(start_app(monkeypatch, model), "some message")
    assert card(at)[0] == "ham"
    assert bars(at) == ["Suspicious: 10.0%", "Ham: 90.0%"]


# ---------- Button, input, repeated use ----------
def test_empty_and_whitespace_input_show_warning(monkeypatch):
    at = start_app(monkeypatch, FakeModel(0.90))
    for text in ("", "   \n  "):
        analyze(at, text)
        assert [w.value for w in at.warning] == ["Please enter a message to analyze."]
        assert card(at) is None


def test_one_click_shows_result_and_keeps_text(monkeypatch):
    at = start_app(monkeypatch, FakeModel(0.90))
    assert card(at) is None  # nothing before the click
    analyze(at, "See you at 6")
    assert card(at) is not None  # the single click was enough
    assert at.text_area[0].value == "See you at 6"


def test_repeated_analysis_updates_result(monkeypatch):
    at = start_app(monkeypatch, FakeModel(0.90))
    analyze(at, "first message")
    first = card(at)
    monkeypatch.setattr(joblib, "load", lambda path: FakeModel(0.10))
    st.cache_resource.clear()
    analyze(at, "second, edited message")
    assert first[0] == "ham"
    assert card(at)[0] == "suspicious"
    assert at.text_area[0].value == "second, edited message"


# ---------- Error handling ----------
def test_missing_model_file_shows_error(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda self: False)
    at = AppTest.from_file(APP_FILE, default_timeout=120).run()
    assert any("Model file not found" in e.value for e in at.error)
    assert len(at.text_area) == 0  # app stopped before drawing the form


def test_unloadable_model_shows_error(monkeypatch):
    def boom(path):
        raise ValueError("incompatible version")

    monkeypatch.setattr(joblib, "load", boom)
    at = AppTest.from_file(APP_FILE, default_timeout=120).run()
    assert any("Could not load the model" in e.value for e in at.error)


def test_wrong_kind_of_model_shows_error(monkeypatch):
    at = start_app(monkeypatch, FakeModel(0.5, classes=("a", "b")))
    assert any("Unexpected model" in e.value for e in at.error)


def test_analysis_error_does_not_echo_the_message(monkeypatch):
    secret = "my-private-text-12345"

    class Exploding(FakeModel):
        def predict_proba(self, texts):
            raise ValueError(f"cannot score: {texts[0]}")

    at = analyze(start_app(monkeypatch, Exploding(0.5)), secret)
    assert any("Something went wrong" in e.value for e in at.error)
    assert not any(secret in e.value for e in at.error)


# ---------- Real model (V4) ----------
def load_v4():
    return REAL_LOAD(V4_FILE)


def test_app_uses_v4_by_default():
    source = Path(APP_FILE).read_text(encoding="utf-8")
    assert "scam_xray_v4.joblib" in source and "scam_xray_v3.joblib" not in source


def test_preprocessing_module_is_available():
    """V4 stores a reference to text_preprocessing.mask_phone_numbers, so it must be importable."""
    assert importlib.util.find_spec("text_preprocessing") is not None
    import text_preprocessing

    assert callable(text_preprocessing.mask_phone_numbers)
    model = load_v4()  # would raise ModuleNotFoundError without the module
    assert model.named_steps["mask"].func is text_preprocessing.mask_phone_numbers


def test_v4_loads_and_predicts():
    model = load_v4()
    assert list(model.classes_) == ["ham", "suspicious"]
    probabilities = model.predict_proba(["Call 09061701461 now to claim your prize", "See you at 6"])
    assert probabilities.shape == (2, 2)
    assert probabilities.sum(axis=1) == pytest.approx([1.0, 1.0])


def test_v4_masks_phone_numbers_inside_the_pipeline():
    """Different phone numbers in otherwise identical messages must score identically."""
    model = load_v4()
    a, b = model.predict_proba(["Call 09061701461 now to claim", "Call 08712345678 now to claim"])
    assert a == pytest.approx(b)


def test_mask_function_behaviour():
    from text_preprocessing import mask_phone_numbers

    assert mask_phone_numbers(["Call 09061701461 now"]) == ["Call NUMBER now"]
    assert mask_phone_numbers(["Call +44 7700 900123 now"]) == ["Call NUMBER now"]
    assert mask_phone_numbers(["Text WIN to 87066"]) == ["Text WIN to 87066"]  # short codes are kept


def test_real_model_in_app(monkeypatch):
    at = start_app(monkeypatch)  # uses the real V4 file
    assert not at.exception and not at.error
    analyze(at, "You won $1000! Click http://bit.ly/x now to claim your prize")
    assert card(at)[0] == "suspicious"
    analyze(at, "See you at 6 for dinner")
    assert card(at)[0] == "ham"
