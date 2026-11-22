import re

from extraction.base import BaseExtractor, ExtractorRegistry
from extraction.helpers import (
    RE_KRS,
    RE_NIP,
    RE_PESEL,
    clean_digits,
    extract_section,
    find_first,
)
from extraction.schemas import (
    MocodawcaData,
    MocodawcaItem,
    PelnomocnictwoSchema,
    PelnomocnikItem,
    PodpisujacyItem,
)


@ExtractorRegistry.register("pelnomocnictwo")
class PelnomocnictwoExtractor(BaseExtractor):
    _RE_PERSON = re.compile(
        r"(?:adw\.|r\.\s*pr\.|radca prawny|adwokat)?\s*([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż-]+)"
    )

    def extract(
        self,
        text_clean: str,
        entities: dict[str, list[str]] | None = None,
    ) -> PelnomocnictwoSchema:
        sec_mocodawca = extract_section(
            text_clean,
            r"mocodawca[:\s]*",
            [r"pełnomocni(?:k|cy)", r"upoważnia", r"udziela"],
        )
        sec_pelnomocnicy = extract_section(
            text_clean,
            r"pełnomocni(?:k|cy)[:\s]*",
            [r"zamawiający", r"postępowanie", r"zakres", r"do\s+reprezentowania"],
        )

        mocodawca_lines = [ln.strip() for ln in sec_mocodawca.splitlines() if ln.strip()]
        nazwa_mocodawcy = mocodawca_lines[0] if mocodawca_lines else None

        mocodawca_item = MocodawcaItem(
            data=MocodawcaData(
                nazwa=nazwa_mocodawcy,
                krs=clean_digits(find_first(RE_KRS, sec_mocodawca or text_clean, group=1)),
                nip=clean_digits(find_first(RE_NIP, sec_mocodawca or text_clean, group=1)),
                typ="osoba_prawna" if re.search(r"\b(sp\.|s\.a\.|krs)\b", sec_mocodawca, re.I) else None,
            ),
            podpisujacy=[PodpisujacyItem(imie_nazwisko=None)],
        )

        pelnomocnicy: list[PelnomocnikItem] = []
        for line in sec_pelnomocnicy.splitlines():
            name = find_first(self._RE_PERSON, line, group=1)
            pesel = find_first(RE_PESEL, line, group=1)
            if name or pesel:
                pelnomocnicy.append(PelnomocnikItem(imie_nazwisko=name, pesel=pesel))

        if not pelnomocnicy:
            pelnomocnicy = [PelnomocnikItem()]

        return PelnomocnictwoSchema(
            mocodawca=[mocodawca_item],
            pelnomocnicy=pelnomocnicy,
        )