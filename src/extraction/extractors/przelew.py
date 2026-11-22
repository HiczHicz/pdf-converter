import re

from extraction.base import BaseExtractor, ExtractorRegistry
from extraction.helpers import (
    RE_AMOUNT,
    RE_DATE,
    RE_IBAN,
    find_all,
    find_first,
    normalize_iban,
    parse_amount,
)
from extraction.schemas import PrzelewBankowy, PrzelewSchema


@ExtractorRegistry.register("przelew", "potwierdzenie_przelewu")
class PrzelewExtractor(BaseExtractor):
    _RE_TX_NUM = re.compile(
        r"(?:numer|nr)\s*(?:transakcji|operacji|referencyjny)[:\s]*([A-Za-z0-9/_-]+)",
        re.IGNORECASE,
    )
    _RE_TITLE = re.compile(
        r"tytuł(?:em)?(?:\s+operacji|\s+przelewu)?[:\s]*(.+)",
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

        amount_raw = find_first(RE_AMOUNT, text_clean, group=1)
        data_operacji = find_first(RE_DATE, text_clean, group=1)
        numer_transakcji = find_first(self._RE_TX_NUM, text_clean, group=1)
        tytul = find_first(self._RE_TITLE, text_clean, group=1)

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