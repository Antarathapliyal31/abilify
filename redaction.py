from functools import lru_cache
from typing import Iterable

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


# Explicit whitelist of entities to redact. Keep tight so that legitimate
# clinical query terms (dates, years, locations, acronyms like "DOB")
# don't get scrubbed and break scope checking or retrieval.
REDACT_ENTITIES: tuple[str, ...] = (
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "MEDICAL_LICENSE",
    "IP_ADDRESS",
    "IBAN_CODE",
    "US_DRIVER_LICENSE",
    "US_PASSPORT",
    "DATE_OF_BIRTH",
)


# A full date (year + month + day) in any common layout. Bare years are
# intentionally NOT matched here — queries like "research after 2016" survive.
# Alternation ordered longest-first so the regex engine never commits to a
# prefix (e.g. "1" instead of "12" for the day).
_DATE_RE = (
    r"(?:"
    r"(?:19|20)\d{2}[-/](?:1[0-2]|0[1-9])[-/](?:3[01]|[12]\d|0[1-9])"   # 1985-03-12
    r"|(?:1[0-2]|0?[1-9])[-/](?:3[01]|[12]\d|0?[1-9])[-/](?:19|20)\d{2}"  # 03/12/1985
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
    r"\s+\d{1,2},?\s+(?:19|20)\d{2}"                                      # March 12, 1985
    r")"
)

# Label words that indicate a date-of-birth context. The entire label+date
# span is matched as one Presidio hit, so anonymization wipes both (no
# dangling "DOB" word left behind). Context-free dates stay untouched.
_DOB_LABEL = (
    r"(?i)(?:"
    r"\bDOB\b"
    r"|\bD\.O\.B\.?"
    r"|\bdate\s+of\s+birth\b"
    r"|\bborn(?:\s+(?:on|in))?\b"
    r"|\bbirth(?:day|date)\b"
    r")"
)

_DOB_PATTERNS = [
    Pattern(
        name="dob_labeled",
        # Allow up to ~15 chars of filler ("is", "was", ":", " - ") between
        # the label and the date; keep it short to avoid pulling unrelated
        # text into the span.
        regex=rf"{_DOB_LABEL}[\s:,\-]{{0,3}}(?:is|was|=)?[\s:,\-]{{0,5}}{_DATE_RE}",
        score=0.6,
    ),
]

# The small spaCy model (en_core_web_sm) tags common psychiatric drug names
# as PERSON (e.g. "Prozac", "Zoloft"). Allow-list protects legitimate drug
# tokens from being redacted. Lowercase; matched case-insensitively by Presidio.
DRUG_ALLOW_LIST: tuple[str, ...] = (
    "abilify", "aripiprazole",
    "prozac", "fluoxetine",
    "zoloft", "sertraline",
    "lexapro", "escitalopram",
    "celexa", "citalopram",
    "paxil", "paroxetine",
    "effexor", "venlafaxine",
    "cymbalta", "duloxetine",
    "wellbutrin", "bupropion",
    "lithium",
    "lamictal", "lamotrigine",
    "depakote", "valproate", "valproic",
    "risperdal", "risperidone",
    "seroquel", "quetiapine",
    "zyprexa", "olanzapine",
    "haldol", "haloperidol",
    "clozaril", "clozapine",
    "latuda", "lurasidone",
    "vraylar", "cariprazine",
    "rexulti", "brexpiprazole",
    "geodon", "ziprasidone",
    "topamax", "topiramate",
    "xanax", "alprazolam",
    "klonopin", "clonazepam",
    "ativan", "lorazepam",
    "valium", "diazepam",
    "adderall", "ritalin",
)


@lru_cache(maxsize=1)
def _analyzer() -> AnalyzerEngine:
    cfg = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    }
    nlp_engine = NlpEngineProvider(nlp_configuration=cfg).create_engine()
    engine = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    engine.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="DATE_OF_BIRTH",
            patterns=_DOB_PATTERNS,
        )
    )
    return engine


@lru_cache(maxsize=1)
def _anonymizer() -> AnonymizerEngine:
    return AnonymizerEngine()


def redact(text: str, entities: Iterable[str] = REDACT_ENTITIES) -> str:
    """Redact PII entities in `text`, replacing each with <ENTITY_TYPE>.

    Returns the input unchanged if no entities are detected.
    """
    if not text or not text.strip():
        return text

    entity_list = list(entities)
    results = _analyzer().analyze(
        text=text,
        language="en",
        entities=entity_list,
        allow_list=[rf"(?i)\b{term}\b" for term in DRUG_ALLOW_LIST],
        allow_list_match="regex",
        # DATE_OF_BIRTH base score is 0.3; context boost (+0.35) pushes it to
        # 0.65 only when "DOB"/"born"/"date of birth" is nearby. Threshold 0.4
        # blocks bare-date false positives while keeping all other recognizers
        # (PERSON ~0.85, EMAIL 1.0, SSN pattern-5 0.5, etc.).
        score_threshold=0.4,
    )
    if not results:
        return text

    operators = {
        e: OperatorConfig("replace", {"new_value": f"<{e}>"})
        for e in entity_list
    }
    anonymized = _anonymizer().anonymize(
        text=text,
        analyzer_results=results,
        operators=operators,
    )
    return anonymized.text
