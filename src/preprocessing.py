"""
Data-cleaning helpers used by the language-model scripts.

Why the quirky '##' / '#' markers?
----------------------------------
The original use case expected special symbols to mark sentence
beginnings and ends so the trigram model can learn that context.
Keeping them here preserves backwards compatibility with the supplied
experiments, while the rest of the code treats them like ordinary
characters.
"""
from typing import List
import re

_DIGIT = re.compile(r"\d")
_BAD = re.compile(r"[^a-z0 .]")      # characters to strip
_END = "#"


def preprocess_line(line: str, *, is_first: bool = False) -> str:
    """
    Clean one raw line of text.

    Steps
    -----
    1. Lower-case
    2. Replace every digit with '0'
    3. Strip accents / punctuation (keep letters, space, '.', '0')
    4. Ensure the line ends with '.'
    5. Append a sentinel '#'
    6. Prepend '##' to the very first line so the model can detect
       document start.

    Returns
    -------
    str
        The cleaned line or ``''`` if the input was blank.
    """
    if not line.strip():
        return ""

    line = line.lower()
    line = _DIGIT.sub("0", line)
    line = _BAD.sub("", line).strip()

    if not line.endswith("."):
        line += "."

    processed = f"{line}{_END}"
    if is_first:
        processed = f"##{processed}"
    return processed


def preprocess_text(lines: List[str]) -> str:
    """Apply :func:`preprocess_line` to every line and join them."""
    cleaned = [
        preprocess_line(line, is_first=(i == 0))
        for i, line in enumerate(lines)
        if line.strip()
    ]
    return " ".join(cleaned)