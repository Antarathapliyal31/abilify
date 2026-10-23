"""Unit tests for the Presidio redaction layer. No external APIs needed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from redaction import redact


def test_redacts_name():
    out = redact("My name is John Smith, does Abilify cause weight gain?")
    assert "<PERSON>" in out
    assert "John Smith" not in out
    assert "Abilify" in out


def test_redacts_email():
    out = redact("Contact me at jane.doe@example.com about Abilify")
    assert "<EMAIL_ADDRESS>" in out
    assert "jane.doe@example.com" not in out


def test_redacts_phone():
    out = redact("Reach me at 415-555-0199 about Abilify")
    assert "<PHONE_NUMBER>" in out
    assert "415-555-0199" not in out


def test_redacts_ssn():
    out = redact("Patient SSN 456-78-1234 was prescribed Abilify")
    assert "<US_SSN>" in out
    assert "456-78-1234" not in out


def test_preserves_drug_names():
    text = "Can I take Abilify with Prozac, Zoloft, or Lexapro?"
    out = redact(text)
    assert "Abilify" in out
    assert "Prozac" in out
    assert "Zoloft" in out
    assert "Lexapro" in out
    assert "<PERSON>" not in out


def test_preserves_dates_and_years():
    out = redact("What research on Abilify was published after 2016?")
    assert out == "What research on Abilify was published after 2016?"


def test_redacts_labeled_dob():
    for text in (
        "DOB 1985-03-12, side effects?",
        "D.O.B.: 03/12/1985",
        "date of birth 03/12/1985",
        "Patient born on March 12, 1985",
        "birthday is 1985-03-12",
    ):
        out = redact(text)
        assert "<DATE_OF_BIRTH>" in out, text
        assert "1985" not in out, text


def test_preserves_non_dob_full_dates():
    for text in (
        "Lot number 2023-04-15 was manufactured for Abilify",
        "Visit on 03/12/2024 for Abilify refill",
    ):
        assert redact(text) == text


def test_noop_on_clean_query():
    text = "What is the mechanism of action of aripiprazole?"
    assert redact(text) == text


def test_empty_input():
    assert redact("") == ""
    assert redact("   ") == "   "


def test_combined_pii_and_drug():
    out = redact(
        "I am Dr. Alice Chen (alice@clinic.com) asking about Abilify + Prozac combo"
    )
    assert "<PERSON>" in out
    assert "Alice Chen" not in out
    assert "<EMAIL_ADDRESS>" in out
    assert "Abilify" in out
    assert "Prozac" in out
