"""tts/normalise.py — deterministic text normalisation for voice output.

Converts visually-rendered text into text that sounds natural when read aloud
by a TTS engine.  No LLM calls — pure regex and rule-based transforms, fast
enough to run inline before every synthesis call.

Entry point: ``normalise_for_speech(text: str) -> str``
"""
from __future__ import annotations

import re


# ── Integer → English words ───────────────────────────────────────────────────

_ONES = [
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety",
]


def _int_to_words(n: int) -> str:
    """Return the English word form of a non-negative integer."""
    if n < 0:
        return "minus " + _int_to_words(-n)
    if n == 0:
        return "zero"
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = _TENS[n // 10], _ONES[n % 10]
        return tens + ("-" + ones if ones else "")
    if n < 1_000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + (" " + _int_to_words(rest) if rest else "")
    if n < 1_000_000:
        rest = n % 1_000
        return _int_to_words(n // 1_000) + " thousand" + (" " + _int_to_words(rest) if rest else "")
    if n < 1_000_000_000:
        rest = n % 1_000_000
        return _int_to_words(n // 1_000_000) + " million" + (" " + _int_to_words(rest) if rest else "")
    if n < 1_000_000_000_000:
        rest = n % 1_000_000_000
        return _int_to_words(n // 1_000_000_000) + " billion" + (" " + _int_to_words(rest) if rest else "")
    rest = n % 1_000_000_000_000
    return _int_to_words(n // 1_000_000_000_000) + " trillion" + (" " + _int_to_words(rest) if rest else "")


def _ordinal_to_words(n: int) -> str:
    """Return the English ordinal word form (e.g. 3 → 'third')."""
    word = _int_to_words(n)
    # Replace the last word of the number with its ordinal form.
    _suffixes: list[tuple[str, str]] = [
        ("one", "first"), ("two", "second"), ("three", "third"), ("four", "fourth"),
        ("five", "fifth"), ("six", "sixth"), ("seven", "seventh"), ("eight", "eighth"),
        ("nine", "ninth"), ("ten", "tenth"), ("eleven", "eleventh"), ("twelve", "twelfth"),
        ("thirteen", "thirteenth"), ("fourteen", "fourteenth"), ("fifteen", "fifteenth"),
        ("sixteen", "sixteenth"), ("seventeen", "seventeenth"), ("eighteen", "eighteenth"),
        ("nineteen", "nineteenth"), ("twenty", "twentieth"), ("thirty", "thirtieth"),
        ("forty", "fortieth"), ("fifty", "fiftieth"), ("sixty", "sixtieth"),
        ("seventy", "seventieth"), ("eighty", "eightieth"), ("ninety", "ninetieth"),
        ("hundred", "hundredth"), ("thousand", "thousandth"),
        ("million", "millionth"), ("billion", "billionth"), ("trillion", "trillionth"),
    ]
    for base, ordinal in _suffixes:
        if word.endswith(base):
            return word[: -len(base)] + ordinal
    return word + "th"


def _decimal_to_words(integer_part: str, frac_part: str) -> str:
    """'3.14' → 'three point one four'"""
    left = _int_to_words(int(integer_part))
    right = " ".join(_int_to_words(int(d)) for d in frac_part)
    return left + " point " + right


# ── Simple fractions ──────────────────────────────────────────────────────────

_SIMPLE_FRACTIONS: dict[tuple[int, int], str] = {
    (1, 2): "one half",
    (1, 3): "one third",
    (2, 3): "two thirds",
    (1, 4): "one quarter",
    (3, 4): "three quarters",
    (1, 8): "one eighth",
    (3, 8): "three eighths",
    (5, 8): "five eighths",
    (7, 8): "seven eighths",
}


def _fraction_to_words(num: int, den: int) -> str:
    if (num, den) in _SIMPLE_FRACTIONS:
        return _SIMPLE_FRACTIONS[(num, den)]
    num_w = _int_to_words(num)
    den_w = _ordinal_to_words(den)
    suffix = "s" if num > 1 else ""
    return f"{num_w} {den_w}{suffix}"


# ── Currency symbols ──────────────────────────────────────────────────────────

_CURRENCY_MAP: dict[str, tuple[str, str]] = {
    "$": ("dollar", "cent"),
    "£": ("pound", "penny"),
    "€": ("euro", "cent"),
    "¥": ("yen", ""),
    "₹": ("rupee", "paisa"),
}


def _currency_to_words(symbol: str, amount: str) -> str:
    major_name, minor_name = _CURRENCY_MAP.get(symbol, ("dollar", "cent"))
    if "." in amount:
        major_str, minor_str = amount.split(".", 1)
        major = int(major_str.replace(",", "") or "0")
        minor_raw = minor_str[:2].ljust(2, "0")
        minor = int(minor_raw)
    else:
        major = int(amount.replace(",", "") or "0")
        minor = 0

    parts: list[str] = []
    if major or not minor:
        s = "" if major == 1 else "s"
        parts.append(_int_to_words(major) + f" {major_name}" + s)
    if minor and minor_name:
        pence_pl = "pence" if major_name == "pound" else (minor_name + ("" if minor == 1 else "s"))
        parts.append(_int_to_words(minor) + " " + pence_pl)
    return " and ".join(parts) if parts else _int_to_words(0) + f" {major_name}s"


# ── Abbreviations ─────────────────────────────────────────────────────────────

# Order matters: longer entries before shorter ones.
_ABBREVS: list[tuple[str, str]] = [
    (r"\be\.g\.", "for example"),
    (r"\bi\.e\.", "that is"),
    (r"\betc\.", "and so on"),
    (r"\bvs\.", "versus"),
    (r"\bvs\b", "versus"),
    (r"\bapprox\.", "approximately"),
    (r"\bAKA\b", "also known as"),
    (r"\baka\b", "also known as"),
    (r"\bw/\b", "with"),
    (r"\bw/o\b", "without"),
    (r"\bAI\b", "A.I."),      # keep it letter-by-letter
    (r"\bAPI\b", "A.P.I."),
    (r"\bURL\b", "U.R.L."),
    (r"\bHTTP[S]?\b", "HTTP"),  # most TTS reads this fine
    (r"\bSQL\b", "S.Q.L."),
    (r"\bUI\b", "U.I."),
    (r"\bUX\b", "U.X."),
    (r"\bOS\b", "O.S."),
    (r"\bRAM\b", "ram"),
    (r"\bCPU\b", "C.P.U."),
    (r"\bGPU\b", "G.P.U."),
]

# Compile once.
_ABBREV_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(p, re.IGNORECASE), repl) for p, repl in _ABBREVS
]


# ── Units ─────────────────────────────────────────────────────────────────────

# Number followed by unit — must come before plain-number expansion.
# Pattern: (unit_regex, singular_expansion, plural_expansion | None)
_UNITS: list[tuple[str, str, str]] = [
    (r"km/h", "kilometre per hour", "kilometres per hour"),
    (r"m/s", "metre per second", "metres per second"),
    (r"mph", "mile per hour", "miles per hour"),
    (r"GHz", "gigahertz", "gigahertz"),
    (r"MHz", "megahertz", "megahertz"),
    (r"kHz", "kilohertz", "kilohertz"),
    (r"Hz", "hertz", "hertz"),
    (r"TB", "terabyte", "terabytes"),
    (r"GB", "gigabyte", "gigabytes"),
    (r"MB", "megabyte", "megabytes"),
    (r"KB", "kilobyte", "kilobytes"),
    (r"kB", "kilobyte", "kilobytes"),
    (r"km", "kilometre", "kilometres"),
    (r"cm", "centimetre", "centimetres"),
    (r"mm", "millimetre", "millimetres"),
    (r"kg", "kilogram", "kilograms"),
    (r"mg", "milligram", "milligrams"),
    (r"ms", "millisecond", "milliseconds"),
    (r"fps", "frames per second", "frames per second"),
    (r"px", "pixel", "pixels"),
]

# °C / °F before number expansion so we don't mangle the number.
_TEMP_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)\s*°\s*([CF])\b")


def _replace_temp(m: re.Match[str]) -> str:
    val = m.group(1)
    unit = "Celsius" if m.group(2).upper() == "C" else "Fahrenheit"
    try:
        n = float(val)
        if n == int(n):
            word = _int_to_words(int(n))
        else:
            word = _decimal_to_words(*val.split(".", 1))
    except ValueError:
        word = val
    return word + " degrees " + unit


# ── Time ──────────────────────────────────────────────────────────────────────

_TIME_PATTERN = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")


def _replace_time(m: re.Match[str]) -> str:
    h, mn = int(m.group(1)), int(m.group(2))
    hour_word = _int_to_words(h)
    if mn == 0:
        return hour_word + " o'clock"
    min_word = _int_to_words(mn) if mn >= 10 else "oh " + _int_to_words(mn)
    return hour_word + " " + min_word


# ── Decade (e.g. "1980s", "80s") ─────────────────────────────────────────────

_DECADE_PATTERN = re.compile(r"\b((?:19|20)?\d0)s\b")

_DECADE_PLURAL: dict[str, str] = {
    "twenty": "twenties", "thirty": "thirties", "forty": "forties",
    "fifty": "fifties", "sixty": "sixties", "seventy": "seventies",
    "eighty": "eighties", "ninety": "nineties", "ten": "tens", "zero": "zeros",
}


def _replace_decade(m: re.Match[str]) -> str:
    n = int(m.group(1))
    # 4-digit years (1900-2099): "1980" → "nineteen eighty", not cardinal form.
    if 1_900 <= n <= 2_099:
        century_word = _int_to_words(n // 100)   # "nineteen" / "twenty"
        decade_word = _int_to_words(n % 100)      # "eighty" / "twenty"
        base_words = [century_word, decade_word]
    else:
        base_words = _int_to_words(n).split()
    last = base_words[-1]
    plural = _DECADE_PLURAL.get(last, last + "s")
    return " ".join(base_words[:-1] + [plural]).strip()


# ── Markdown stripping ────────────────────────────────────────────────────────

def _strip_markdown(text: str) -> str:
    # Fenced code blocks → label only.
    text = re.sub(r"```[a-zA-Z]*\n[\s\S]*?```", "[code block]", text)
    # Inline code.
    text = re.sub(r"`[^`]+`", lambda m: m.group(0)[1:-1], text)
    # ATX headings (#, ##, ...).
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Bold/italic.
    text = re.sub(r"\*{1,3}([^*]+?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+?)_{1,3}", r"\1", text)
    # Strikethrough.
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    # Links — keep display text.
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Bare URLs.
    text = re.sub(r"https?://\S+", "a link", text)
    # Blockquote markers.
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)
    # Unordered list bullets (-, *, +).
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    # Ordered list markers (1. 2. etc).
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    # Horizontal rules.
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    return text


# ── Special punctuation ───────────────────────────────────────────────────────

def _normalise_punctuation(text: str) -> str:
    # Em-dash, en-dash → comma-space (pause).
    text = text.replace("—", ", ")
    text = text.replace("–", " to ")
    # Ellipsis.
    text = text.replace("…", ", ")
    text = re.sub(r"\.{2,}", ", ", text)
    # Ampersand between words → "and".
    text = re.sub(r"\b&\b", "and", text)
    text = text.replace(" & ", " and ")
    # Tilde used for approximation.
    text = re.sub(r"~(\d)", r"approximately \1", text)
    # Plus sign between words/numbers → "plus".
    text = re.sub(r"(?<=\w)\s*\+\s*(?=\w)", " plus ", text)
    # Remove remaining stray symbols unlikely to be spoken.
    text = re.sub(r"[|\\^<>{}]", " ", text)
    return text


# ── Main normalisation pipeline ───────────────────────────────────────────────

def normalise_for_speech(text: str) -> str:
    """Transform *text* into a form that reads naturally when spoken aloud.

    Transformations (in order):
    1. Strip markdown.
    2. Normalise punctuation / special symbols.
    3. Expand abbreviations.
    4. Temperatures (°C/°F) — before number expansion.
    5. Time patterns (HH:MM).
    6. Decades ("1980s" → "nineteen eighties").
    7. Currency ($, £, €, ¥).
    8. Ordinal numbers (1st, 2nd …).
    9. Percentages (75%).
    10. Simple fractions (3/4).
    11. Numbers with thousands-separators (40,075).
    12. Decimal numbers (3.14).
    13. Plain integers.
    14. Units (km/h, GB, ms …).
    15. Final whitespace cleanup.
    """
    if not text:
        return text

    text = _strip_markdown(text)
    text = _normalise_punctuation(text)

    # Abbreviations.
    for pattern, replacement in _ABBREV_PATTERNS:
        text = pattern.sub(replacement, text)

    # Temperatures.
    text = _TEMP_PATTERN.sub(_replace_temp, text)

    # Time (24-h and 12-h style "14:30", "9:00").
    text = _TIME_PATTERN.sub(_replace_time, text)

    # Decades ("1980s", "80s", "2020s").
    text = _DECADE_PATTERN.sub(_replace_decade, text)

    # Currency — match symbol then optional space then digits.
    def _replace_currency(m: re.Match[str]) -> str:
        return _currency_to_words(m.group(1), m.group(2))

    text = re.sub(
        r"([£$€¥₹])\s*([\d,]+(?:\.\d+)?)",
        _replace_currency,
        text,
    )

    # Ordinals (1st, 2nd, 3rd, 4th … 999th).
    def _replace_ordinal(m: re.Match[str]) -> str:
        return _ordinal_to_words(int(m.group(1)))

    text = re.sub(r"\b(\d{1,4})(st|nd|rd|th)\b", _replace_ordinal, text)

    # Percentages.
    def _replace_pct(m: re.Match[str]) -> str:
        num_str = m.group(1).replace(",", "")
        if "." in num_str:
            ip, fp = num_str.split(".", 1)
            word = _decimal_to_words(ip, fp)
        else:
            word = _int_to_words(int(num_str))
        return word + " percent"

    text = re.sub(r"\b([\d,]+(?:\.\d+)?)\s*%", _replace_pct, text)

    # Simple fractions (must come before plain-number expansion).
    def _replace_fraction(m: re.Match[str]) -> str:
        num, den = int(m.group(1)), int(m.group(2))
        if den == 0:
            return m.group(0)
        return _fraction_to_words(num, den)

    text = re.sub(r"\b(\d{1,3})/(\d{1,3})\b", _replace_fraction, text)

    # Numbers with thousands separators (e.g. 40,075 or 1,000,000).
    def _replace_comma_number(m: re.Match[str]) -> str:
        raw = m.group(0).replace(",", "")
        try:
            return _int_to_words(int(raw))
        except ValueError:
            return m.group(0)

    text = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", _replace_comma_number, text)

    # Decimal numbers (not preceded by a digit to avoid re-matching).
    def _replace_decimal(m: re.Match[str]) -> str:
        ip, fp = m.group(1), m.group(2)
        try:
            return _decimal_to_words(ip, fp)
        except ValueError:
            return m.group(0)

    text = re.sub(r"\b(\d+)\.(\d+)\b", _replace_decimal, text)

    # Plain integers (4+ digits to avoid changing everyday small numbers the
    # TTS engine already reads correctly, e.g. years and model numbers).
    # We DO expand ≥ 4-digit standalone integers and any ≥ 1000.
    def _replace_plain_int(m: re.Match[str]) -> str:
        try:
            n = int(m.group(0))
            if n >= 1000:
                return _int_to_words(n)
        except ValueError:
            pass
        return m.group(0)

    text = re.sub(r"\b\d+\b", _replace_plain_int, text)

    # Units (number-less occurrences — just the label, e.g. "in GB" or "per km/h").
    for unit_re, singular, plural in _UNITS:
        # After a digit word ("gigabytes") already fine; catch bare label occurrences.
        text = re.sub(
            rf"(?<!\w){re.escape(unit_re)}\b",
            plural,
            text,
        )

    # Collapse multiple spaces / newlines.
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r" {2,}", " ", text)
    text = text.strip()

    return text
