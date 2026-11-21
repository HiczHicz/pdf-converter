import re

_MONTHS = r"stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|września|października|listopada|grudnia"

ENTITY_PATTERNS: dict[str, re.Pattern[str]] = {
    # Przelew: rachunki 26-cyfrowe (z PL lub bez)
    "TOKEN_IBAN": re.compile(r"\bP[LI1|]?\s?(\d{2}(?:\s?\d{4}){6})\b", re.IGNORECASE),
    
    # Zamówienia publiczne (silny sygnał dla Odwołania KIO)
    "TOKEN_BZP": re.compile(r"\b\d{4}/BZP\s?\d{8}(?:/\d{2})?\b", re.IGNORECASE),
    "TOKEN_TED": re.compile(r"\b(?:\d{6}-\d{4}|\d{4}/S\s?\d{3}-\d{6})\b", re.IGNORECASE),
    
    # Identyfikatory podmiotów (z kontekstem, żeby odróżnić 10-cyfrowy KRS od NIP)
    "TOKEN_KRS": re.compile(r"\bKRS[:\s]*\d{10}\b", re.IGNORECASE),
    "TOKEN_NIP": re.compile(r"\bNIP[:\s]*(?:PL)?\s?\d{3}[-\s]?\d{3}[-\s]?\d{2}[-\s]?\d{2}\b|\bNIP[:\s]*(?:PL)?\s?\d{3}[-\s]?\d{2}[-\s]?\d{2}[-\s]?\d{3}\b", re.IGNORECASE),
    "TOKEN_REGON": re.compile(r"\bREGON[:\s]*\d{9}(?:\d{5})?\b", re.IGNORECASE),
    "TOKEN_PESEL": re.compile(r"\bPESEL[:\s]*\d{11}\b", re.IGNORECASE),
    
    # Dane kontaktowe i finansowe
    "TOKEN_EMAIL": re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"),
    "TOKEN_DATE": re.compile(
    rf"\b(?:\d{{2}}[./-]\d{{2}}[./-]\d{{4}}|\d{{4}}[./-]\d{{2}}[./-]\d{{2}}|\d{{1,2}}\s+(?:{_MONTHS})\s+\d{{4}}(?:\s+r\.|\s+roku)?)\b",
    re.IGNORECASE,
),
    "TOKEN_AMOUNT": re.compile(r"\b\d{1,3}(?:[\s.]\d{3})*(?:[.,]\d{2})\s?(?:PLN|zł|EUR|USD)?\b", re.IGNORECASE),

    "TOKEN_POSTAL": re.compile(r"\b\d{2}-\d{3}\b"),
    "TOKEN_PHONE": re.compile(r"\btel\.?\s*(?:\+48\s?)?(?:\d[\s-]?){9}\b", re.IGNORECASE),
}

BOILERPLATE_PATTERNS: list[re.Pattern[str]] = [
    #re.compile(r"Strona\s+\d+\s+z\s+\d+", re.IGNORECASE),
]