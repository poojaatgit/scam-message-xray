# 🔍 Scam Message X-Ray

A small Streamlit web app that estimates whether an SMS message looks **Ham** (normal), **Suspicious**, or **Uncertain**, using a TF-IDF + Logistic Regression model.

> **Disclaimer:** This is an educational machine-learning portfolio project. Predictions are probability estimates and are **not guaranteed to be accurate**. Do not rely on it for security decisions.

**Live demo:** _coming soon (URL to be added)_

## Features

- Paste an SMS message and click **Analyze Message** once to see the result.
- Three outcomes: **Ham**, **Suspicious**, or **Uncertain — Further verification needed**.
- Always shows the model's Ham and Suspicious probabilities with progress bars.
- Uncertain results include a reminder to verify through an official channel.
- Handles empty input and model-loading errors with clear messages.
- Inference only: the app loads a pre-trained model and never trains or changes it.

## Tech stack

| Tool | Purpose |
|---|---|
| Python 3.11–3.13 | Language (tested on 3.13) |
| Streamlit | Web interface |
| scikit-learn 1.6.1 | The model's library (TF-IDF + Logistic Regression) |
| joblib | Loading the saved model |

## Model and training approach

The model (`scam_xray_v4.joblib`, "V4") was trained **separately in Google Colab**. The training notebook is not part of this repository; this repo only contains the trained model for inference.

V4 uses the same training data, splits, TF-IDF settings and Logistic Regression settings as the earlier V3 model. The intended change was **number masking before feature extraction**, for privacy.

Model structure, read from the saved file (a scikit-learn pipeline):

1. **Masking:** `text_preprocessing.mask_phone_numbers` replaces phone-number-like strings (7 or more digits, optionally with spaces, dots, dashes or brackets) with the word `NUMBER`. This happens **inside the model**, so the app passes messages through unchanged.
2. **Features:** two TF-IDF vectorizers combined side by side
   - word n-grams (1–2 words), `min_df=2`, sublinear term frequency
   - character n-grams (3–5 characters), `min_df=2`, sublinear term frequency
   - about 75,700 features in total
3. **Classifier:** Logistic Regression (`class_weight="balanced"`, `max_iter=1500`, `random_state=42`)
4. **Classes:** `ham` and `suspicious`

The saved model refers to the masking function by name, so `text_preprocessing.py` must stay in the project folder, and **changing that file changes how V4 behaves**.

**Datasets:** see [Data sources and attribution](#data-sources-and-attribution) below. How many messages went into training versus evaluation is not documented here.

## Data sources and attribution

The model was trained using material from two datasets published on Mendeley Data under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license (license status checked on each dataset's page):

1. **SMS Phishing Dataset.** Mishra, S., & Soni, D. (2022). *SMS Phishing Dataset for Machine Learning and Pattern Recognition* [Data set]. Mendeley Data, V1. [https://doi.org/10.17632/f45bkkt8pr.1](https://doi.org/10.17632/f45bkkt8pr.1)
   - **Size (per the publisher):** 5,971 messages: 4,844 ham, 489 spam and 638 smishing.
   - **Publisher's requests and acknowledgements:** the authors ask users to cite their paper of the same title (the dataset page gives no venue or DOI for it). They acknowledge Tiago A. Almeida and Dr. Gunikhan Sonowal for making text messages available, and list as related datasets the SMS Spam Collection and a "Smishing dataset". They also list related publications, for example Mishra, S., & Soni, D. (2020), *Smishing Detector: A security model to detect smishing through SMS content analysis and URL behavior analysis*, Future Generation Computer Systems, [https://doi.org/10.1016/j.future.2020.03.021](https://doi.org/10.1016/j.future.2020.03.021).
   - **Provenance:** the dataset page does not say which messages come from those acknowledged sources, and this README makes no claim about it.
2. **Financial Scams Detection Dataset.** Ahmed, A. R., & Islam, G. F. (2025). *Financial Scams Detection Dataset* [Data set]. Mendeley Data, V1. [https://doi.org/10.17632/znsk27yk3h.1](https://doi.org/10.17632/znsk27yk3h.1)
   - **Size:** the publisher's page does not state a message count. The downloaded file contains 523 labeled messages (310 scam, 213 ham).
   - **Consent and anonymization (publisher's description, not independently verified here):** the publisher describes the messages as collected with the senders' consent, with names, phone numbers, account numbers and e-mail addresses removed or masked.

**Related upstream resource (not claimed to be included):** the SMS Spam Collection by Almeida & Hidalgo (2011), UCI Machine Learning Repository, [https://doi.org/10.24432/C5CC84](https://doi.org/10.24432/C5CC84), whose UCI page states CC BY 4.0. It is acknowledged here because the SMS Phishing Dataset's authors list it as related, not because this project confirmed it is part of that dataset.

**Changes made:** the model was trained on both datasets (the original SMS training data and the financial-scam development training data). The data was cleaned, and the datasets' labels were mapped into this project's two classes before training:

| Original label | Project label |
|---|---|
| Ham | `ham` |
| Spam | `suspicious` |
| Smishing | `suspicious` |
| Scam | `suspicious` |

Merging several scam-like labels into one `suspicious` class means the model does not distinguish spam from smishing or scams. The trained model is a derived work; the original authors have not reviewed or endorsed it or this project.

The datasets themselves are not included in this repository.

**Licensing of the model:** CC BY 4.0 allows sharing and adapting material, including for commercial purposes, as long as credit is given, the license is linked, and changes are indicated. This project does that above. Whether a trained model counts as "adapted material" is not settled by the license text, so this repository takes the conservative route of applying those same attribution terms to the model file. Two points are **not verified**: the license of the separate "Smishing dataset" that the SMS dataset's authors list as related (it is linked from a Pinterest page), and whether any of its messages are in the SMS Phishing Dataset. This is not legal advice.

**Privacy note:** TF-IDF models store a vocabulary, so the model file contains words and short fragments learned from the training messages (only terms that appear in at least two messages). An earlier model (V3) stored about 50 complete phone-number-like strings from spam examples. **V4 was trained with number masking, and a check of its stored vocabulary found no run of 7 or more digits.** Shorter codes (such as 5-digit SMS short codes) and ordinary words from the training messages can still appear in it. Because the masking rule is broad, dates and long amounts written with 7 or more digits may also be replaced by `NUMBER`. V3 is kept only as a private local backup and is not distributed. If you believe something in the model concerns you, please open an issue.

## Evaluation results and limitations

**These are development evaluation results, not independent real-world validation.** They were measured separately from this app, in the training environment.

| Evaluation (V4) | Messages | Accuracy | Suspicious recall | Macro-F1 |
|---|---|---|---|---|
| Original SMS test set | 870 | 98.97% | 95.92% | 98.14% |
| Financial development set | 101 | 91.09% | 98.31% | 90.56% |

Confusion matrices (rows are the true class, columns are the prediction; order: `ham`, `suspicious`):

| Original SMS test set | Predicted ham | Predicted suspicious |
|---|---|---|
| **True ham** | 720 | 3 |
| **True suspicious** | 6 | 141 |

| Financial development set | Predicted ham | Predicted suspicious |
|---|---|---|
| **True ham** | 34 | 8 |
| **True suspicious** | 1 | 58 |

For comparison, V3 scored 98.85% accuracy and 95.24% suspicious recall on the SMS test set, and 90.10% accuracy, 98.31% suspicious recall and 89.45% macro-F1 on the financial set.

How to read these numbers:

- **The original SMS test set was reused during model development.** It was looked at more than once while the model was being built and improved, so its scores are likely **optimistic** and are not a clean, untouched test.
- **The financial evaluation is a small development set** (101 messages). A few messages more or fewer would move the percentages noticeably.
- On the financial set, 8 of 42 normal messages were flagged as suspicious, while only 1 of 59 suspicious messages was missed. The model leans towards flagging, which is the reason for the Uncertain result.
- **These results are not guaranteed real-world performance.** No final, untouched test set has been evaluated, and the figures are not produced by this repository's code or tests.

Known limitations:

- The model only sees the message text, not the sender, links' destinations, or context.
- It reflects the data it was trained on. Messages in other languages, styles, or newer scam tactics may be misjudged.
- The probabilities are the model's raw estimates; they have not been checked for calibration.
- The 20-point Uncertain threshold is a **provisional starting value**. It has not been validated on held-out data and may change. Uncertain results mean "ask for independent verification", not "safe".
- It is a learning project, not a security product.

## How results are decided

The app reads the Ham and Suspicious probabilities (looked up by class name from `model.classes_`) and compares them:

| Gap between the two probabilities | Result |
|---|---|
| 20 percentage points or more | The higher class: **Ham** or **Suspicious** |
| Less than 20 percentage points | **Uncertain — Further verification needed** |

Examples: 90% Ham / 10% Suspicious → **Ham**. 3% / 97% → **Suspicious**. Exactly 60% / 40% (a 20-point gap) → **Ham**. 55% / 45% (a 10-point gap) → **Uncertain**.

The threshold is `UNCERTAIN_THRESHOLD_PP` at the top of [app.py](app.py). It is provisional.

## Install and run

Requires **Python 3.11 to 3.13**. Python 3.14 is not supported yet because the pinned scikit-learn 1.6.1 has no wheels for it. The dependency set was tested on Python 3.13 (Windows); 3.11 and 3.12 are expected to work because all pinned packages publish wheels for them, but they have not been tested.

```bash
git clone <your-repository-url>
cd scam-message-xray
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows (PowerShell)
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install and start the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 if the browser does not open automatically.

The direct dependencies are pinned in `requirements.txt` (Streamlit 1.66.1, scikit-learn 1.6.1, joblib 1.6.0, NumPy 2.2.6, SciPy 1.15.3). `scikit-learn` must stay at **1.6.1**, the version that saved the model; other versions may fail to load it or behave differently. Packages that Streamlit itself installs (pandas, pyarrow and others) are not pinned.

## Tests

The tests use Streamlit's built-in test tool and need `pytest` (not required to run the app):

```bash
pip install -r requirements-dev.txt
python -m pytest
```

They cover Ham, Suspicious and Uncertain results, the 20-point boundary, empty input, repeated analysis, model-loading errors, and the V4 model itself (it loads, the preprocessing module is available, and phone numbers are masked inside the model).

## Project structure

```
scam-message-xray/
├── app.py                  # Streamlit app
├── scam_xray_v4.joblib     # Pre-trained model (loaded, never modified)
├── text_preprocessing.py   # Number-masking function that the model needs
├── requirements.txt        # Pinned dependencies
├── requirements-dev.txt    # Adds pytest for the tests
├── tests/                  # UI and model tests
├── README.md
└── .gitignore
```

## Security note

`.joblib` files are Python pickles and can run code when loaded. Only load model files you trust. The model in this repository is the author's own, and it also runs `text_preprocessing.py` from this repository.

## Copyright, Educational Purpose & Usage

**Educational purpose.** This project was developed solely for educational, learning, research experimentation, and personal portfolio demonstration purposes. It is not intended for commercial use or production deployment.

**Copyright.** Copyright © 2026. All rights reserved. The original source code in this repository (`app.py`, `text_preprocessing.py` and the files in `tests/`) is copyrighted. No permission is granted to reproduce, modify, redistribute, or commercially use the original source code without prior written authorization from the copyright holder. This repository does not carry an open-source license.

This repository is public so that it can be viewed as a portfolio piece. Public visibility does not grant any further rights: any use of the code is limited to what the law and GitHub's Terms of Service allow. This notice is a statement of the author's rights, not a technical restriction.

**Experimental predictions.** The machine learning predictions are experimental. They should not be relied upon as a definitive assessment of whether a message is safe or fraudulent.

**Third-party materials.** Third-party libraries, datasets, and other materials remain subject to their respective licenses and terms. The training datasets are published by their authors under CC BY 4.0, are not included in this repository, and are not covered by the copyright statement above. The trained model file `scam_xray_v4.joblib` was derived from them and is distributed with the attribution given in [Data sources and attribution](#data-sources-and-attribution).
