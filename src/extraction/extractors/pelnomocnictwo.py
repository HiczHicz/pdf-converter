import re

from extraction.base import BaseExtractor, ExtractorRegistry
from extraction.helpers import (
    RE_KRS,
    RE_NIP,
    RE_PESEL,
    clean_digits,
    extract_section,
    find_all,
    find_first,
)
from extraction.schemas import (
    MocodawcaData,
    MocodawcaItem,
    PelnomocnictwoSchema,
    PelnomocnikItem,
    PodpisujacyItem,
)

_NAME_RE = re.compile(
    r"\b([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż-]+)\b"
)
_TITLED_PROXY_RE = re.compile(
    r"(?i:\b(?:r\.\s*pr\.|adw\.|radca\s+prawny|adwokat))\s+"
    r"([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż-]+)"
)
_ROLE_RE = re.compile(
    r"\b(?:prezes\s+zarządu|wiceprezes(?:\s+zarządu)?|członek\s+zarządu|prokurent|dyrektor|wspólnik|komplementariusz|właściciel|pełnomocnik\s+zarządu)\b",
    re.IGNORECASE,
)
_LEGAL_ENTITY_RE = re.compile(
    r"\b(?:sp\.\s*z\s*o\.o\.|sp\.\s*k\.|sp\.\s*j\.|sp\.\s*p\.|s\.a\.|spółka|fundacja|stowarzyszenie|instytut|uniwersytet|gmina|urząd|szpital|filharmonia|krs)\b",
    re.IGNORECASE,
)
_PHYSICAL_ENTITY_RE = re.compile(
    r"\b(?:osoba\s+fizyczna|działalność\s+gospodarcz|prowadząc\w*\s+działalność|zamieszkał\w*|legitymując\w*\s+się|pesel)\b",
    re.IGNORECASE,
)
_BAD_STRINGS = {
    "pełnomocnik", "pełnomocnicy", "pełnomocnictwo", "mocodawca", "mocodawcy",
    "zamawiający", "postępowanie", "dane rejestrowe", "prezes zarządu",
    "członek zarządu", "wiceprezes zarządu", "krajowej izby", "izby odwoławczej",
    "pełnomocnictwo przyjmuję", "miejsce na", "pieczęć firmową",
}


@ExtractorRegistry.register("pelnomocnictwo")
class PelnomocnictwoExtractor(BaseExtractor):
    @staticmethod
    def _clean_company_name(raw: str | None) -> str | None:
        if not raw:
            return None
        name = re.sub(
            r"^(?:\d+[.)]\s*)?(?:lider(?:\s+konsorcjum)?|partner|mocodawcą\s+jest|mocodawcami\s+są|mocodawca(?:\s*\d+)?(?:\s+to)?|działając\s+w\s+imieniu|reprezentując)\s*[:,-]?\s*",
            "",
            raw.strip(),
            flags=re.I,
        )
        name = re.sub(r"\s*\(dalej.*$", "", name, flags=re.I)
        name = re.sub(
            r",?\s*\b(?:ul\.|ulica|al\.|pl\.|os\.|z\s+siedzibą|siedziba|KRS|NIP|REGON|PESEL|reprezentowan\w*|prowadząc\w*)\b.*$",
            "",
            name,
            flags=re.I,
        )
        name = " ".join(name.split()).strip(' ,.;:-"')
        if not name or name.lower() in _BAD_STRINGS:
            return None
        return name

    def _split_mocodawca_blocks(self, text_clean: str) -> list[str]:
        sec = extract_section(
            text_clean,
            r"(?:§\s*1\.?\s*)?mocodawc(?:a|y)[:\s]*",
            [r"§\s*2", r"(?m)^\s*pełnomocni(?:k|cy)\b", r"ustanawia(?:m|ją)?\s+pełnomocnik", r"upoważnia(?:m|ją)?", r"udziela(?:m|ją)?\s+pełnomocnictwa"],
        )
        if not sec:
            return [text_clean]

        # Podział na wielu mocodawców (konsorcjum: "1) ... 2) ...", "1. ... 2. ..." lub "Mocodawca 1 / 2")
        parts = re.split(
            r"(?m)^\s*(?=(?:\d+[.)]\s+|(?:Mocodawca\s*\d+|Lider\s+konsorcjum|Partner\s+konsorcjum)\s*[:,-]))",
            sec,
            flags=re.I,
        )
        parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 10]
        if len(parts) > 1:
            return parts
        return [sec]

    def _parse_mocodawca_data(self, block: str, full_text: str, is_single: bool) -> MocodawcaData:
        krs = clean_digits(find_first(RE_KRS, block, 1))
        nip = clean_digits(find_first(RE_NIP, block, 1))
        if is_single:
            krs = krs or clean_digits(find_first(RE_KRS, full_text, 1))
            nip = nip or clean_digits(find_first(RE_NIP, full_text, 1))

        first_valid = None
        for ln in block.splitlines():
            cleaned = self._clean_company_name(ln)
            if cleaned:
                first_valid = cleaned
                break

        if not first_valid and is_single:
            m = re.search(
                r"(?:Mocodawcą\s+jest|Działając\s+w\s+imieniu)\s+([^\n]+)",
                full_text,
                re.I,
            )
            if m:
                first_valid = self._clean_company_name(m.group(1))

        is_legal = bool(krs or _LEGAL_ENTITY_RE.search(first_valid or "") or _LEGAL_ENTITY_RE.search(block))
        is_physical = (not is_legal) and bool(
            _PHYSICAL_ENTITY_RE.search(block) or (first_valid and _NAME_RE.fullmatch(first_valid))
        )

        if is_physical:
            person_name = find_first(_NAME_RE, block, 1)
            jdg_match = re.search(r'pod\s+firmą\s+["„]?([^"”\n,]+)["”]?', block, re.I)
            firm_name = self._clean_company_name(jdg_match.group(1)) if jdg_match else None
            return MocodawcaData(
                imie_nazwisko=person_name,
                krs=None,
                nazwa=firm_name,
                nip=nip,
                typ="osoba_fizyczna",
            )

        return MocodawcaData(
            imie_nazwisko=None,
            krs=krs,
            nazwa=first_valid,
            nip=nip,
            typ="osoba_prawna" if (is_legal or first_valid) else None,
        )

    def _extract_signatories(self, text_clean: str) -> list[str]:
        lines = text_clean.splitlines()
        found: list[str] = []

        # 1. Nazwiska bezpośrednio nad rolami (obsługuje 2 kolumny i wiele osób obok siebie)
        for idx, line in enumerate(lines):
            if not _ROLE_RE.search(line):
                continue
            for offset in (1, 2):
                if idx - offset < 0:
                    continue
                prev = lines[idx - offset].strip()
                if not prev or re.fullmatch(r"[._\s-]+", prev):
                    continue
                # Usuń prawą kolumnę z pełnomocnikiem (r. pr. / adw.)
                prev = re.split(r"\b(?:r\.\s*pr\.|adw\.|radca\s+prawny|adwokat)\b", prev, maxsplit=1, flags=re.I)[0]
                names = [n for n in _NAME_RE.findall(prev) if n.lower() not in _BAD_STRINGS]
                if names:
                    for n in names:
                        if n not in found:
                            found.append(n)
                    break

        # 2. Fallback dla podpisu bez roli pod spodem (np. osoba fizyczna pod linią kropek na końcu dokumentu)
        if not found:
            tail = lines[int(len(lines) * 0.65):]
            for idx, line in enumerate(tail):
                if re.search(r"\.{5,}|_{5,}", line) and idx + 1 < len(tail):
                    nxt = re.split(r"\b(?:r\.\s*pr\.|adw\.|radca\s+prawny|adwokat)\b", tail[idx + 1], maxsplit=1, flags=re.I)[0]
                    for n in _NAME_RE.findall(nxt):
                        if n.lower() not in _BAD_STRINGS and n not in found:
                            found.append(n)

        return found

    def _extract_pelnomocnicy(self, text_clean: str) -> list[PelnomocnikItem]:
        sec_proxy = extract_section(
                    text_clean,
                    r"(?:§\s*2\.?\s*Pełnomocni(?:k|cy)|^\s*Pełnomocni(?:k|cy)|ustanawia(?:m|ją)?\s+pełnomocnikiem)[:\s]*",
                    [r"§\s*3", r"^\s*Zamawiający\b", r"^\s*Postępowanie\b", r"^\s*Zakres\b", r"do\s+reprezentowania"],
                )
        scope = re.sub(r"\s+", " ", sec_proxy or text_clean)

        matches = list(_TITLED_PROXY_RE.finditer(scope))
        if not matches and sec_proxy:
            matches = [
                m for m in _NAME_RE.finditer(scope)
                if m.group(1).lower() not in _BAD_STRINGS
            ]

        seen: set[str] = set()
        items: list[PelnomocnikItem] = []
        for idx, m in enumerate(matches):
            name = m.group(1).strip()
            if name in seen or name.lower() in _BAD_STRINGS:
                continue
            seen.add(name)

            # Szukaj PESEL-a w oknie kontekstu między tym a kolejnym pełnomocnikiem
            win_end = matches[idx + 1].start() if idx + 1 < len(matches) else min(len(scope), m.end() + 140)
            pesel = find_first(RE_PESEL, scope[m.end():win_end], 1)
            items.append(PelnomocnikItem(imie_nazwisko=name, pesel=pesel))

        if not items:
            pesels = find_all(RE_PESEL, scope, 1)
            items = [PelnomocnikItem(pesel=p) for p in pesels] if pesels else [PelnomocnikItem()]

        return items

    def extract(
        self,
        text_clean: str,
        entities: dict[str, list[str]] | None = None,
    ) -> PelnomocnictwoSchema:
        blocks = self._split_mocodawca_blocks(text_clean)
        is_single = len(blocks) == 1
        signatories = self._extract_signatories(text_clean)

        mocodawcy: list[MocodawcaItem] = []
        for idx, block in enumerate(blocks):
            data = self._parse_mocodawca_data(block, text_clean, is_single=is_single)
            if is_single:
                podpisujacy = (
                    [PodpisujacyItem(imie_nazwisko=s) for s in signatories]
                    if signatories
                    else [PodpisujacyItem()]
                )
            else:
                # Przy wielu mocodawcach rozdzielamy podpisujących równomiernie po kolejności bloków
                chunk = max(1, len(signatories) // len(blocks)) if signatories else 0
                start = idx * chunk
                end = len(signatories) if idx == len(blocks) - 1 else start + chunk
                assigned = signatories[start:end] if chunk else []
                podpisujacy = (
                    [PodpisujacyItem(imie_nazwisko=s) for s in assigned]
                    if assigned
                    else [PodpisujacyItem()]
                )

            mocodawcy.append(MocodawcaItem(data=data, podpisujacy=podpisujacy))

        return PelnomocnictwoSchema(
            mocodawca=mocodawcy,
            pelnomocnicy=self._extract_pelnomocnicy(text_clean),
        )