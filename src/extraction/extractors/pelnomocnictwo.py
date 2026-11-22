import re

from extraction.base import BaseExtractor, ExtractorRegistry
from extraction.helpers import (
    RE_KRS,
    RE_NIP,
    RE_PESEL,
    clean_digits,
    extract_section,
    find_first,
    find_all,
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
    _RE_SIGNATORY = re.compile(
        r"(?m)^([A-ZĄĆĘŁŃÓŚŹŻ][A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż-]+(?:\s+[A-ZĄĆĘŁŃÓŚŹŻ][A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż-]+){1,3})"
        r"\s*(?:\r?\n|\s*$)",
    )
    _RE_SIGNATORY_ROLE = re.compile(
        r"^\s*(?:prezes zarządu|prokurent|dyrektor|członek zarządu|wiceprezes)\s*$",
        re.IGNORECASE | re.MULTILINE,
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
        proxy_source = sec_pelnomocnicy
        if not proxy_source:
            proxy_source = "\n".join(
                line for line in text_clean.splitlines()
                if re.search(r"\b(?:adw\.|r\.\s*pr\.|radca prawny|adwokat)\b", line, re.I)
            )

        mocodawca_lines = [ln.strip() for ln in sec_mocodawca.splitlines() if ln.strip()]
        nazwa_mocodawcy = mocodawca_lines[0] if mocodawca_lines else None
        if not nazwa_mocodawcy:
            fallback_name = re.search(
                r"(?:Mocodawcą jest|Mocodawca(?:\s+to)?|Działając w imieniu)\s+"
                r"([^,\n]+(?:sp\.?\s*(?:z\s*o\.o\.|k\.|k\.a\.)|s\.a\.|S\.A\.)?)",
                text_clean,
                re.IGNORECASE,
            )
            if fallback_name:
                nazwa_mocodawcy = fallback_name.group(1).strip()
        if nazwa_mocodawcy:
            nazwa_mocodawcy = re.sub(
                r"\b(?:ul\.|ulica|KRS\s*:?|NIP\s*:?|REGON\s*:?|siedziba|\(dalej\s*:)\b.*$",
                "",
                nazwa_mocodawcy,
                flags=re.IGNORECASE,
            ).strip(" ,")
        if nazwa_mocodawcy and nazwa_mocodawcy.lower().startswith("mocodawcą jest"):
            nazwa_mocodawcy = re.sub(r"^mocodawcą jest\s*", "", nazwa_mocodawcy, flags=re.I)
        if nazwa_mocodawcy and nazwa_mocodawcy.lower().startswith("mocodawca"):
            nazwa_mocodawcy = re.sub(r"^mocodawca\s*", "", nazwa_mocodawcy, flags=re.I)
        if nazwa_mocodawcy and "," in nazwa_mocodawcy:
            nazwa_mocodawcy = nazwa_mocodawcy.split(",", 1)[0].strip()

        mocodawca_item = MocodawcaItem(
            data=MocodawcaData(
                nazwa=nazwa_mocodawcy,
                krs=clean_digits(find_first(RE_KRS, sec_mocodawca or text_clean, group=1)),
                nip=clean_digits(find_first(RE_NIP, sec_mocodawca or text_clean, group=1)),
                typ="osoba_prawna" if re.search(r"\b(?:sp\.|s\.a\.|krs)\b", sec_mocodawca or text_clean, re.I) else None,
            ),
            podpisujacy=[PodpisujacyItem(imie_nazwisko=None)],
        )

        signatory_name = None
        direct_signatory = re.search(
            r"(?m)^([A-ZĄĆĘŁŃÓŚŹŻ][A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż-]+"
            r"(?:\s+[A-ZĄĆĘŁŃÓŚŹŻ][A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż-]+){1,3})"
            r"\s*\r?\n\s*(?:prezes\s+zarządu|prokurent|dyrektor|członek\s+zarządu|wiceprezes)\s*$",
            text_clean,
            re.IGNORECASE,
        )
        if direct_signatory:
            signatory_name = direct_signatory.group(1).strip()
        if not self._RE_SIGNATORY_ROLE.search(text_clean):
            role_match = None
        else:
            role_match = self._RE_SIGNATORY_ROLE.search(text_clean)
        if role_match and not signatory_name:
            role_start = role_match.start()
            prefix = text_clean[max(0, role_start - 160):role_start]
            prefix_lines = [line.strip() for line in prefix.splitlines() if line.strip()]
            for line in reversed(prefix_lines):
                candidate = re.fullmatch(
                    r"[A-ZĄĆĘŁŃÓŚŹŻ][A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż-]+"
                    r"(?:\s+[A-ZĄĆĘŁŃÓŚŹŻ][A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż-]+){1,3}",
                    line,
                )
                if candidate:
                    signatory_name = candidate.group(0)
                    break
        mocodawca_item.podpisujacy = [PodpisujacyItem(imie_nazwisko=signatory_name)]

        pelnomocnicy: list[PelnomocnikItem] = []
        for line in proxy_source.splitlines():
            names = find_all(self._RE_PERSON, line, group=1)
            pesels = find_all(RE_PESEL, line, group=1)
            if names:
                for index, name in enumerate(names):
                    pesel = pesels[index] if index < len(pesels) else None
                    pelnomocnicy.append(PelnomocnikItem(imie_nazwisko=name, pesel=pesel))
            elif pesels:
                for pesel in pesels:
                    pelnomocnicy.append(PelnomocnikItem(pesel=pesel))

        if not pelnomocnicy:
            pelnomocnicy = [PelnomocnikItem()]

        return PelnomocnictwoSchema(
            mocodawca=[mocodawca_item],
            pelnomocnicy=pelnomocnicy,
        )