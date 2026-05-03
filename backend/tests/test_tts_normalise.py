"""Tests for tts/normalise.py — voice text normalisation."""
import pytest

from tts.normalise import normalise_for_speech


# ── Numbers with comma separators ─────────────────────────────────────────────

def test_comma_number_five_digits():
    assert normalise_for_speech("40,075 km") == "forty thousand seventy-five kilometres"

def test_comma_number_one_million():
    assert normalise_for_speech("1,000,000 people") == "one million people"

def test_comma_number_complex():
    assert normalise_for_speech("The distance is 12,345 metres.") == \
        "The distance is twelve thousand three hundred forty-five metres."

def test_large_number_no_commas():
    assert normalise_for_speech("Population: 8000000") == "Population: eight million"


# ── Decimal numbers ───────────────────────────────────────────────────────────

def test_decimal_basic():
    assert normalise_for_speech("3.14") == "three point one four"

def test_decimal_zero_prefix():
    assert normalise_for_speech("0.5") == "zero point five"

def test_decimal_in_sentence():
    result = normalise_for_speech("Pi is approximately 3.14159.")
    assert "three point one four one five nine" in result


# ── Percentages ───────────────────────────────────────────────────────────────

def test_percentage_integer():
    assert normalise_for_speech("75%") == "seventy-five percent"

def test_percentage_decimal():
    result = normalise_for_speech("3.5% interest rate")
    assert "three point five percent" in result


# ── Ordinals ──────────────────────────────────────────────────────────────────

def test_ordinal_1st():
    assert normalise_for_speech("1st place") == "first place"

def test_ordinal_2nd():
    assert normalise_for_speech("2nd attempt") == "second attempt"

def test_ordinal_3rd():
    assert normalise_for_speech("3rd floor") == "third floor"

def test_ordinal_4th():
    assert normalise_for_speech("4th time") == "fourth time"

def test_ordinal_21st():
    assert normalise_for_speech("21st century") == "twenty-first century"


# ── Currency ──────────────────────────────────────────────────────────────────

def test_currency_dollars_cents():
    assert normalise_for_speech("$4.99") == "four dollars and ninety-nine cents"

def test_currency_whole_dollars():
    assert normalise_for_speech("$100") == "one hundred dollars"

def test_currency_pounds():
    result = normalise_for_speech("£5")
    assert "five pound" in result

def test_currency_euros():
    result = normalise_for_speech("€10.50")
    assert "ten euro" in result and "fifty cent" in result


# ── Temperature ───────────────────────────────────────────────────────────────

def test_temperature_celsius():
    assert normalise_for_speech("22°C") == "twenty-two degrees Celsius"

def test_temperature_fahrenheit():
    assert normalise_for_speech("98.6°F") == "ninety-eight point six degrees Fahrenheit"

def test_temperature_negative():
    assert normalise_for_speech("-5°C") == "minus five degrees Celsius"


# ── Time ──────────────────────────────────────────────────────────────────────

def test_time_on_the_hour():
    assert normalise_for_speech("Meeting at 09:00.") == "Meeting at nine o'clock."

def test_time_with_minutes():
    result = normalise_for_speech("14:30")
    assert result == "fourteen thirty"

def test_time_with_oh_minutes():
    result = normalise_for_speech("9:05")
    assert result == "nine oh five"


# ── Decades ───────────────────────────────────────────────────────────────────

def test_decade_1980s():
    assert normalise_for_speech("1980s music") == "nineteen eighties music"

def test_decade_short_form():
    result = normalise_for_speech("80s hits")
    assert "eighties" in result


# ── Fractions ─────────────────────────────────────────────────────────────────

def test_fraction_three_quarters():
    assert normalise_for_speech("3/4 done") == "three quarters done"

def test_fraction_one_half():
    assert normalise_for_speech("1/2") == "one half"

def test_fraction_arbitrary():
    result = normalise_for_speech("5/6")
    assert "five" in result and "sixth" in result


# ── Markdown stripping ────────────────────────────────────────────────────────

def test_strip_bold():
    assert normalise_for_speech("**important** point") == "important point"

def test_strip_heading():
    result = normalise_for_speech("## Summary\nHere it is.")
    assert "##" not in result
    assert "Summary" in result

def test_strip_bullets():
    result = normalise_for_speech("- First item\n- Second item")
    assert "-" not in result
    assert "First item" in result

def test_strip_inline_code():
    result = normalise_for_speech("Use the `print()` function")
    assert "`" not in result
    assert "print()" in result

def test_strip_url():
    result = normalise_for_speech("See https://example.com for details")
    assert "https://" not in result


# ── Punctuation normalisation ─────────────────────────────────────────────────

def test_em_dash():
    result = normalise_for_speech("good—great")
    assert "—" not in result
    assert "good" in result and "great" in result

def test_ellipsis():
    result = normalise_for_speech("wait...")
    assert "..." not in result

def test_ampersand():
    assert normalise_for_speech("cats & dogs") == "cats and dogs"


# ── Abbreviations ─────────────────────────────────────────────────────────────

def test_abbrev_eg():
    result = normalise_for_speech("e.g. cats")
    assert "for example" in result

def test_abbrev_ie():
    result = normalise_for_speech("i.e. correct")
    assert "that is" in result

def test_abbrev_vs():
    result = normalise_for_speech("cats vs. dogs")
    assert "versus" in result


# ── Units ─────────────────────────────────────────────────────────────────────

def test_unit_kmh_in_sentence():
    result = normalise_for_speech("Speed is 40,075 km/h")
    assert "forty thousand seventy-five" in result
    assert "kilometre" in result

def test_unit_gb():
    result = normalise_for_speech("Storage: 512 GB")
    # After number expansion "512" stays as-is (<1000 but >=1, let engine handle)
    assert "gigabyte" in result

def test_unit_ms():
    result = normalise_for_speech("Latency: 1200 ms")
    assert "millisecond" in result


# ── End-to-end sentence ───────────────────────────────────────────────────────

def test_full_sentence():
    inp = "The circumference of Earth is **40,075 km** (approximately 24,901 miles)."
    result = normalise_for_speech(inp)
    assert "forty thousand seventy-five" in result
    assert "twenty-four thousand nine hundred" in result
    assert "**" not in result
