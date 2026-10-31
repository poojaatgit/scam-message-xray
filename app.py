"""Scam Message X-Ray: a small Streamlit app around a pre-trained SMS classifier.

The model (scam_xray_v4.joblib) was trained separately in Google Colab.
This app only loads it and runs predictions; it never trains or modifies it.
The model masks phone-number-like strings itself (see text_preprocessing.py),
so the app passes the message to it unchanged.
"""

from pathlib import Path

import joblib
import streamlit as st

MODEL_PATH = Path(__file__).parent / "scam_xray_v4.joblib"

# If the Ham and Suspicious probabilities are closer than this many percentage
# points, the result is "Uncertain". PROVISIONAL: not yet validated on held-out data.
UNCERTAIN_THRESHOLD_PP = 20

MAX_MESSAGE_CHARS = 1600  # roughly ten concatenated SMS segments

# Text and CSS class for each outcome. The detail line for "uncertain" has no
# probability because neither class clearly wins.
OUTCOMES = {
    "ham": ("✅ Ham (looks normal)", "Model-estimated probability: <b>{p:.1%}</b>"),
    "suspicious": ("⚠️ Suspicious", "Model-estimated probability: <b>{p:.1%}</b>"),
    "uncertain": (
        "❓ Uncertain — Further verification needed",
        "The model could not clearly separate Ham from Suspicious.",
    ),
}

st.set_page_config(page_title="Scam Message X-Ray", page_icon="🔍", layout="centered")

# CSS for the result cards. Only fixed text from OUTCOMES is inserted into the
# card HTML below, never user input, so unsafe_allow_html is safe here.
st.markdown(
    """
    <style>
    .result-card {
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        border: 1px solid;
    }
    .result-card h3 { margin: 0 0 0.25rem 0; }
    .result-card p  { margin: 0; }
    .ham        { background: rgba(46, 160, 67, 0.12);  border-color: #2ea043; }
    .suspicious { background: rgba(218, 54, 51, 0.12);  border-color: #da3633; }
    .uncertain  { background: rgba(139, 148, 158, 0.15); border-color: #8b949e; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource  # load once and reuse; exceptions are not cached, so a fixed file loads on the next run
def load_model():
    """Load and sanity-check the model. Raises RuntimeError with a readable message on failure."""
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH.name}. Place it next to app.py.")
    try:
        # The saved model refers to text_preprocessing.mask_phone_numbers by name,
        # so that module must be importable before the model is loaded.
        import text_preprocessing  # noqa: F401

        model = joblib.load(MODEL_PATH)
    except Exception as exc:  # e.g. missing module, scikit-learn version mismatch, corrupt file
        raise RuntimeError(f"Could not load the model: {exc}") from exc

    classes = list(getattr(model, "classes_", []))
    if not hasattr(model, "predict_proba") or not {"ham", "suspicious"} <= set(classes):
        raise RuntimeError(
            f"Unexpected model: expected predict_proba() and classes 'ham' and 'suspicious', got {classes}."
        )
    return model


def get_probabilities(model, message):
    """Return (p_ham, p_suspicious), looked up by class name, not by column position."""
    probabilities = model.predict_proba([message])[0]
    by_class = dict(zip(model.classes_, probabilities))
    return float(by_class["ham"]), float(by_class["suspicious"])


def decide(p_ham, p_suspicious, threshold_pp=UNCERTAIN_THRESHOLD_PP):
    """Return 'ham', 'suspicious' or 'uncertain' from two probabilities between 0 and 1."""
    # Rounding avoids float noise: 0.6 - 0.4 is 0.19999999999999996 in Python.
    difference_pp = round(abs(p_ham - p_suspicious) * 100, 6)
    if difference_pp < threshold_pp:
        return "uncertain"
    return "ham" if p_ham > p_suspicious else "suspicious"


def show_result(p_ham, p_suspicious):
    """Render the result card, the Uncertain warning (if any) and both probability bars."""
    outcome = decide(p_ham, p_suspicious)
    title, detail = OUTCOMES[outcome]
    winning_p = p_ham if outcome == "ham" else p_suspicious
    st.markdown(
        f'<div class="result-card {outcome}"><h3>{title}</h3><p>{detail.format(p=winning_p)}</p></div>',
        unsafe_allow_html=True,
    )

    if outcome == "uncertain":
        st.warning(
            "This message sits close to the model's decision boundary. "
            "Please verify it independently through an official channel "
            "(for example, the organisation's official website or phone number), "
            "not through any link or number in the message."
        )

    st.write("")
    st.caption("Probability breakdown")
    st.progress(p_suspicious, text=f"Suspicious: {p_suspicious:.1%}")
    st.progress(p_ham, text=f"Ham: {p_ham:.1%}")


# ---------- Page ----------
st.title("🔍 Scam Message X-Ray")
st.write("Paste an SMS message below and the model will estimate whether it looks suspicious.")

try:
    model = load_model()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()  # nothing else can work without a model

# A form sends the text and the button click to Streamlit together, so one
# click is enough. The text stays in the box after submitting.
with st.form("analyze_form"):
    message = st.text_area(
        "SMS message",
        height=150,
        max_chars=MAX_MESSAGE_CHARS,
        placeholder="e.g. Congratulations! You won a free prize. Click the link to claim now.",
    )
    submitted = st.form_submit_button("Analyze Message", type="primary", width="stretch")

if submitted:
    if not message.strip():
        st.warning("Please enter a message to analyze.")
    else:
        try:
            p_ham, p_suspicious = get_probabilities(model, message)
        except Exception as exc:
            # Show only the error type so the submitted message is never echoed back.
            st.error(f"Something went wrong while analyzing the message ({type(exc).__name__}).")
        else:
            show_result(p_ham, p_suspicious)

st.divider()
st.caption(
    "⚠️ Disclaimer: this is an educational machine-learning project. "
    "Predictions are probability estimates and are not guaranteed to be accurate. "
    "The Uncertain threshold is provisional and has not been validated. "
    "Do not rely on this tool for security decisions."
)
