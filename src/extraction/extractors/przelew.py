import re

from extraction.base import BaseExtractor, ExtractorRegistry
from extraction.helpers import (
    RE_AMOUNT,
    RE_DATE,
    RE_IBAN,
    find_all,
    find_first,
    normalize_iban,
    normalize_polish_date,
    parse_amount,
)
from extraction.schemas import PrzelewBankowy, PrzelewSchema


@ExtractorRegistry.register("przelew", "potwierdzenie_przelewu")
class PrzelewExtractor(BaseExtractor):
    _RE_TRANSACTION_LABEL = re.compile(
        r"(?:numer\s+(?:referencyjny|transakcji|operacji)|nr\s*(?:ref\.?|transakcji|operacji))"
        r"(?:\s+(?:transakcji|operacji|przelewu))?\s*[.:]?\s*"
        r"((?=[A-Za-z0-9/_-]*\d)[A-Za-z0-9/_-]{5,})",
        re.IGNORECASE,
    )
    _RE_TITLE = re.compile(
        r"(?m)^\s*(?:tytuł(?:em)?(?:\s+operacji|\s+przelewu)?|opis)\s*[:.]?\s*"
        r"([^\n]*(?:\n(?!\s*(?:nr\s*ref|numer\s+ref|numer\s+transakcji|status|kanał|typ\s+operacji|tytuł|zrealizowano|wykonano|potwierdzenie\s+ma|data\s+|kwota\s+))[^\n]*)*)",
        re.IGNORECASE,
    )
    _RE_TITLE_CUTOFF = re.compile(
        r"\b(?:kwota\s+(?:operacji|przelewu|transakcji)|data\s+(?:transakcji|operacji|realizacji)"
        r"|nr\s*ref\.?|numer\s+(?:referencyjny|transakcji)|status|zrealizowano|wykonano"
        r"|kanał|typ\s+operacji|potwierdzenie\s+ma)(?=\s|:|$)",
        re.IGNORECASE,
    )
    _RE_OPERATION_DATE = re.compile(
        r"(?:data\s+(?:operacji|realizacji|transakcji)|kwota\s+operacji[^\n]*\n[^\n]*?)"
        r"[^\d\n]{0,20}"
        r"(\d{4}[./-]\d{1,2}[./-]\d{1,2}|\d{1,2}[./-]\d{1,2}[./-]\d{4}|\d{1,2}\s+[A-Za-ząćęłńóśźż]+\s+\d{4})",
        re.IGNORECASE,
    )
    _RE_AMOUNT_LABELED = re.compile(
        r"kwota(?:\s+(?:operacji|przelewu|transakcji))?(?:[^\d\n]*\n)?[^\d\n]{0,20}"
        r"(\d+(?:[\s\u00a0]\d{3})*[.,]\d{2})(?![\d./-])",
        re.IGNORECASE,
    )

    def extract(
        self,
        text_clean: str,
        entities: dict[str, list[str]] | None = None,
    ) -> PrzelewSchema:
        ibans = [normalize_iban(x) for x in find_all(RE_IBAN, text_clean)]
        z_rachunku = ibans[0] if len(ibans) >= 1 else None
        na_rachunek = ibans[1] if len(ibans) >= 2 else None

        amount_raw = (
            find_first(RE_AMOUNT, text_clean, group=1)
            or find_first(self._RE_AMOUNT_LABELED, text_clean, group=1)
        )
        date_match = find_first(self._RE_OPERATION_DATE, text_clean, group=1)
        data_operacji = normalize_polish_date(date_match) or normalize_polish_date(
            find_first(RE_DATE, text_clean, group=1)
        )
        numer_transakcji = find_first(self._RE_TRANSACTION_LABEL, text_clean, group=1)

        title_match = self._RE_TITLE.search(text_clean)
        tytul = None
        if title_match:
            raw_title = " ".join(title_match.group(1).split())
            tytul = self._RE_TITLE_CUTOFF.split(raw_title, maxsplit=1)[0].strip(" :.-") or None

        return PrzelewSchema(
            przelew_bankowy=PrzelewBankowy(
                data_operacji=data_operacji,
                kwota=parse_amount(amount_raw),
                na_rachunek=na_rachunek,
                numer_transakcji=numer_transakcji,
                tytul=tytul,
                z_rachunku=z_rachunku,
            )
        )