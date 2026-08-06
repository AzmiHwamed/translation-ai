import re


def clean_json(text):

    text = re.sub(r"```json", "", text)

    text = re.sub(r"```", "", text)

    return text.strip()
