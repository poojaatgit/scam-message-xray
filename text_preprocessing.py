
import re

# IMPORTANT: this function is part of the saved model, scam_xray_v4.joblib.
# The model stores only its name ("text_preprocessing.mask_phone_numbers"), so
# this file must stay importable under this exact module and function name for
# the model to load. Changing the logic below changes how V4 behaves.
def mask_phone_numbers(texts):
    result = []

    pattern = re.compile(
        r"\+?\d(?:[\s().-]*\d){6,}"
    )

    for text in texts:
        cleaned = pattern.sub("NUMBER", str(text))
        result.append(cleaned)

    return result
