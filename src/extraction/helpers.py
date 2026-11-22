import re

RE_NIP = re.compile(r"\bNIP[:\s]*(?:PL)?\s*([\d-]{10,13})\b", re.IGNORECASE)
RE_KRS = re.compile(r"\bKRS[:\s]*(\d{10})\b", re.IGNORECASE)
RE_REGON = re.compile(r"\bREGON[:\s]*(\d{9}(?:\d{5})?)\b", re.IGNORECASE)
RE_PESEL = re.compile(r"\bPESEL[:\s]*(\d{11})\b", re.IGNORECASE)
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
RE_IBAN = re.compile(r"\b(?:PL\s*)?(?:\d{2}[\s-]?)(?:\d{4}[\s-]?){6}\b")
RE_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{2}[./-]\d{2}[./-]\d{4})\b")
RE_AMOUNT = re.compile(r"\b(\d{1,3}(?:[\s\u00a0]\d{3})*(?:[.,]\d{2}))\s*(?:PLN|zł|ZŁ)\b")
RE_BZP = re.compile(r"\b\d{4}/BZP\s*\d+(?:/\d+)*\b", re.IGNORECASE)
RE_TED = re.compile(r"\b\d{4}/S\s*\d{3}-\d{6}\b|\b\d{6}-\d{4}\b", re.IGNORECASE)

POLISH_MONTHS = {
    "stycznia": "01",
    "lutego": "02",
    "marca": "03",
    "kwietnia": "04",
    "maja": "05",
    "czerwca": "06",
    "lipca": "07",
    "sierpnia": "08",
    "września": "09",
    "października": "10",
    "listopada": "11",
    "grudnia": "12",
}


def normalize_polish_date(value: str | None) -> str | None:
    if not value:
        return None
    value = " ".join(value.strip().split())
    match = re.search(
        r"(?<!\d)(\d{1,2})\s+([A-Za-ząćęłńóśźżĄĆĘŁŃÓŚŹŻ]+)\s+(\d{4})(?!\d)",
        value,
    )
    if match:
        month = POLISH_MONTHS.get(match.group(2).lower())
        if month:
            return f"{match.group(3)}-{month}-{int(match.group(1)):02d}"
    match = re.search(r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)", value)
    if match:
        return f"{match.group(1)}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"
    match = re.search(r"(?<!\d)(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})(?!\d)", value)
    if match:
        return f"{match.group(3)}-{int(match.group(2)):02d}-{int(match.group(1)):02d}"
    return None


def find_first(pattern: re.Pattern[str], text: str, group: int = 0) -> str | None:
    m = pattern.search(text)
    return m.group(group).strip() if m else None


def find_all(pattern: re.Pattern[str], text: str, group: int = 0) -> list[str]:
    return [m.group(group).strip() for m in pattern.finditer(text)]


def extract_section(
    text: str,
    start_pattern: str,
    end_patterns: list[str] | None = None,
) -> str:
    """Wycina fragment tekstu od start_pattern do pierwszego dopasowanego end_pattern."""
    start_match = re.search(start_pattern, text, re.IGNORECASE | re.MULTILINE)
    if not start_match:
        return ""
    rest = text[start_match.end():]
    if not end_patterns:
        return rest.strip()

    end_pos = len(rest)
    for ep in end_patterns:
        end_match = re.search(ep, rest, re.IGNORECASE | re.MULTILINE)
        if end_match and end_match.start() < end_pos:
            end_pos = end_match.start()
    return rest[:end_pos].strip()


def clean_digits(value: str | None) -> str | None:
    if not value:
        return None
    return re.sub(r"\D", "", value) or None


def normalize_iban(value: str | None) -> str | None:
    if not value:
        return None
    normalized = re.sub(r"[\s-]+", "", value).upper()
    return re.sub(r"^PL(?=\d{2})", "", normalized)


def parse_amount(value: str | None) -> float | None:
    if not value:
        return None
    cleaned = value.replace("\u00a0", "").replace(" ", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None