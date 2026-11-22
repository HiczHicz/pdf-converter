import re

from extraction.base import BaseExtractor, ExtractorRegistry
from extraction.helpers import (
    RE_BZP,
    RE_EMAIL,
    RE_KRS,
    RE_NIP,
    RE_REGON,
    clean_digits,
    extract_section,
    find_first,
    normalize_polish_date,
)
from extraction.schemas import (
    OdwolanieKIOData,
    OdwolanieSchema,
    OdwolujacyItem,
    OdwolujacyPelnomocnik,
    ZamawiajacyItem,
    ZamowieniePubliczneData,
)

CZYNNOSC_POWOD_DICT: dict[int, str] = {
    1: "Decyzja o odrzuceniu oferty",
    2: "Ogłoszenie wyboru oferty",
    3: "Decyzja o wykluczeniu wykonawcy",
    4: "Publikacja specyfikacji warunków zamówienia (SWZ)",
    5: "Modyfikacja treści SWZ",
    6: "Odpowiedzi/wyjaśnienia do SWZ",
    7: "Zaniechanie odrzucenia oferty",
    8: "Zaniechanie wykluczenia wykonawcy",
    9: "Decyzja o unieważnieniu postępowania",
    10: "Zaniechanie unieważnienia postępowania",
    11: "Opis przedmiotu zamówienia (OPZ)",
    12: "Projektowane postanowienia umowy",
}

# Kolejność ma znaczenie: zaniechania i modyfikacje przed ogólnymi decyzjami/publikacją
_ACTION_RULES: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"zaniecha\w*\s+unieważnien|nie\s+unieważni\w*\s+postępowan", re.I), 10),
    (re.compile(r"zaniecha\w*\s+odrzucen|nie\s+odrzuci\w*\s+ofert", re.I), 7),
    (re.compile(r"zaniecha\w*\s+wykluczen|nie\s+wykluczy\w*\s+wykonawc", re.I), 8),
    (re.compile(r"unieważni\w*\s+(?:postępowan|rozstrzygnięc)|o\s+unieważnieniu\s+postępowan", re.I), 9),
    (re.compile(r"odrzuceni\w*\s+ofert|odrzuci\w*\s+ofert", re.I), 1),
    (re.compile(r"wybor\w*\s+(?:najkorzystniejszej\s+)?ofert|ogłosi\w*\s+wybór\s+ofert|wybra\w*\s+ofert", re.I), 2),
    (re.compile(r"wykluczeni\w*\s+(?:wykonawc|odwołując|z\s+postępowan)|wykluczy\w*\s+(?:wykonawc|odwołując)", re.I), 3),
    (re.compile(r"zmodyfikowa\w*[^\n.]{0,50}(?:SWZ|specyfikacj)|modyfikacj\w*[^\n.]{0,30}(?:SWZ|specyfikacj)|zmian\w*\s+treści\s+(?:SWZ|specyfikacj)", re.I), 5),
    (re.compile(r"(?:odpowiedz|wyjaśnien)\w*[^\n.]{0,60}(?:SWZ|specyfikacj|pytań|pytania)", re.I), 6),
    (re.compile(r"projektowan\w*\s+postanowie\w*\s+umow|\bPPU\b", re.I), 12),
    (re.compile(r"opis\w*\s+przedmiotu\s+zamówieni|\bOPZ\b", re.I), 11),
    (re.compile(r"(?:opublikowa\w*|publikacj\w*|udostępni\w*|zamieści\w*)[^\n.]{0,50}(?:SWZ|specyfikacj\w*\s+warunków\s+zamówieni)", re.I), 4),
)

_DATE_TOKEN = (
    r"(?:\d{4}[./-]\d{1,2}[./-]\d{1,2}"
    r"|\d{1,2}[./-]\d{1,2}[./-]\d{4}"
    r"|\d{1,2}\s+(?:stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|września|października|listopada|grudnia)\s+\d{4})"
)

_RE_PROXY_PERSON = re.compile(
    r"(?i:\b(?:r\.\s*pr\.|adw\.|radca\s+prawny|adwokat))\s+"
    r"([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż-]+)"
)
_RE_PROC_NUM = re.compile(
    r"(?:nr\s+referencyjny\s+sprawy|numer\s+referencyjny\s+sprawy|nr\s+referencyjny|numer\s+referencyjny"
    r"|nr\s+postępowania|numer\s+postępowania|nr\s+sprawy|znak\s+sprawy)"
    r"\s*[:.]?\s*((?=[A-Za-z0-9./_-]*\d)[A-Za-z0-9./_-]+)",
    re.IGNORECASE,
)


@ExtractorRegistry.register("odwolanie")
class OdwolanieExtractor(BaseExtractor):
    def _extract_header_city_date(self, text_clean: str) -> tuple[str | None, str | None]:
        # Szukamy wyłącznie w nagłówku (przed treścią merytoryczną / zarzutami)
        cutoff = re.search(r"Działając\s+w\s+imieniu|Zarzut\s+nr\s*1", text_clean, re.I)
        header = text_clean[: cutoff.start()] if cutoff else text_clean[:1500]

        city_date = re.search(
            rf"(?m)^\s*([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż-]+(?:\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż-]+)?)"
            rf"(?:\s*,\s*(?:dnia\s+)?|\s+(?:dnia\s+)?)"
            rf"({_DATE_TOKEN})(?:\s*(?:r\.|roku))?\s*$",
            header,
        )
        if city_date:
            city = city_date.group(1).strip()
            if not re.search(r"\b(?:ul|al|pl|strona|dnia)\b", city, re.I):
                return city, normalize_polish_date(city_date.group(2))

        standalone_date = re.search(rf"(?m)^\s*(?:dnia\s+)?({_DATE_TOKEN})(?:\s*(?:r\.|roku))?\s*$", header)
        if standalone_date:
            return None, normalize_polish_date(standalone_date.group(1))

        return None, None

    def _extract_action(self, text_clean: str) -> tuple[str | None, int | None, str | None]:
        # Ucinamy tekst przed "Wnoszę o:" (żądania), żeby nie łapać tamtejszych fraz
        body_end = re.search(r"(?m)^\s*Wnoszę\s+o\s*:", text_clean, re.I)
        body = text_clean[: body_end.start()] if body_end else text_clean
        flat_body = re.sub(r"\s+", " ", body)

        # 1. Zdanie o zaskarżonej czynności ("Zamawiający w dniu ... ...")
        action_sent = re.search(
            rf"Zamawiający\b[^.]{{0,120}}?(?:w\s+dniu|dnia)\s+({_DATE_TOKEN})(?:\s*(?:r\.|roku))?[^.]{{3,260}}\.",
            flat_body,
            re.IGNORECASE,
        )
        action_dzien = normalize_polish_date(action_sent.group(1)) if action_sent else None
        action_text = action_sent.group(0) if action_sent else ""

        # 2. Blok pierwszego zarzutu (wyłącznie do Zarzutu nr 2)
        first_zarzut = extract_section(
            body,
            r"Zarzut\s+(?:nr\s*)?1\b",
            [r"Zarzut\s+(?:nr\s*)?2\b", r"(?m)^\s*Wnoszę\s+o\s*:"],
        )
        flat_zarzut = re.sub(r"\s+", " ", first_zarzut)

        if not action_dzien and flat_zarzut:
            m_date = re.search(rf"(?:w\s+dniu|dnia|z\s+dnia)\s+({_DATE_TOKEN})", flat_zarzut, re.I)
            if m_date:
                action_dzien = normalize_polish_date(m_date.group(1))

        # Dopasowanie powodu: najpierw zdanie o czynności, potem pierwszy zarzut
        powod_id: int | None = None
        for scope in (action_text, flat_zarzut):
            if not scope:
                continue
            for pattern, cid in _ACTION_RULES:
                if pattern.search(scope):
                    powod_id = cid
                    break
            if powod_id is not None:
                break

        opis = CZYNNOSC_POWOD_DICT.get(powod_id) if powod_id is not None else None
        return action_dzien, powod_id, opis

    @staticmethod
    def _clean_entity_name(raw: str | None) -> str | None:
        if not raw:
            return None
        name = re.sub(
            r"^(?:lider(?:\s+konsorcjum)?|partner|wykonawca|odwołujący|zamawiający)\s*[:,-]?\s*",
            "",
            raw.strip(),
            flags=re.I,
        )
        name = re.sub(r"\s*\(dalej.*$", "", name, flags=re.I)
        name = re.sub(r",?\s*\b(?:ul\.|al\.|pl\.|os\.|KRS|NIP|REGON)\b.*$", "", name, flags=re.I)
        return " ".join(name.split()).strip(" ,.;:-") or None

    def _extract_odwolujacy(self, text_clean: str) -> list[OdwolujacyItem]:
        sec_odw = extract_section(
            text_clean,
            r"(?m)^\s*Odwołujący\s*[:\s]*",
            [r"(?m)^\s*(?:\d\)\s*)?Zamawiający\b", r"(?m)^\s*dotyczy\s*:", r"(?m)^\s*ODWOŁANIE\b"],
        )
        if not sec_odw:
            return [OdwolujacyItem()]

        # Jeśli konsorcjum, ograniczamy dane podmiotu do Lidera
        leader_sec = sec_odw
        leader_m = re.search(r"lider(?:\s+konsorcjum)?\s*[:,-]?\s*", sec_odw, re.I)
        if leader_m:
            partner_m = re.search(r"(?:partner|członek\s+konsorcjum|2\))", sec_odw[leader_m.end():], re.I)
            end = leader_m.end() + partner_m.start() if partner_m else len(sec_odw)
            leader_sec = sec_odw[leader_m.end():end]

        lines = [ln.strip() for ln in leader_sec.splitlines() if ln.strip()]
        nazwa = self._clean_entity_name(lines[0]) if lines else None

        has_proxy = bool(re.search(r"reprezentowan|pełnomocni|\badw\.|\br\.\s*pr\.", sec_odw, re.I))
        flat_odw = re.sub(r"\s+", " ", sec_odw)

        proxy_person = find_first(_RE_PROXY_PERSON, flat_odw, group=1) if has_proxy else None
        proxy_mail = find_first(RE_EMAIL, flat_odw) if has_proxy else None

        proxy_firm = None
        if has_proxy:
            firm_m = re.search(
                r"(?:adres\s+do\s+korespondencji\s*:\s*)?"
                r"(Kancelaria[^,\n]+?(?:odpowiedzialnością|Sp\.\s*z\s*o\.o\.|sp\.\s*k\.|sp\.\s*p\.|i\s+Partnerzy|Radców\s+Prawnych|Adwokacka|Prawna)?)"
                r"(?=\s*,\s*(?:ul\.|al\.|pl\.|os\.|\d{2}-\d{3})|\s*tel\.|\s*$)",
                flat_odw,
                re.I,
            )
            if firm_m:
                proxy_firm = " ".join(firm_m.group(1).split()).strip(" ,.;")

        return [
            OdwolujacyItem(
                nazwa=nazwa,
                krs=clean_digits(find_first(RE_KRS, leader_sec, 1)),
                nip=clean_digits(find_first(RE_NIP, leader_sec, 1)),
                regon=clean_digits(find_first(RE_REGON, leader_sec, 1)),
                przez_pelnomocnika=1 if has_proxy else 0,
                pelnomocnik=OdwolujacyPelnomocnik(
                    imie_nazwisko=proxy_person,
                    nazwa=proxy_firm,
                    mail=proxy_mail,
                ),
            )
        ]

    def _extract_zamawiajacy(self, text_clean: str) -> list[ZamawiajacyItem]:
        # Obsługa wielokrotnych Zamawiających (Lider / Partner lub numerowane bloki)
        multi_matches = list(
            re.finditer(
                r"(?m)^\s*(?:\d+\)\s*)?Zamawiający(?:\s*[-–]\s*(?:Lider|Partner|prowadzący\s+postępowanie)[^\n:]*)?\s*:",
                text_clean,
                re.I,
            )
        )
        end_markers = [r"(?m)^\s*Odwołujący\s*:", r"(?m)^\s*dotyczy\s*:", r"(?m)^\s*ODWOŁANIE\b", r"(?m)^\s*Postępowanie\s*:"]

        blocks: list[tuple[str, str]] = []
        if len(multi_matches) > 1:
            for idx, m in enumerate(multi_matches):
                start = m.end()
                next_start = multi_matches[idx + 1].start() if idx + 1 < len(multi_matches) else len(text_clean)
                chunk = text_clean[start:next_start]
                for ep in end_markers:
                    em = re.search(ep, chunk, re.I)
                    if em:
                        chunk = chunk[: em.start()]
                blocks.append((m.group(0), chunk.strip()))
        else:
            sec_zam = extract_section(text_clean, r"(?m)^\s*Zamawiający\s*[:\s]*", end_markers)
            if sec_zam:
                # Sprawdź podział wewnętrzny 1) ... 2) ... w obrębie jednej sekcji Zamawiający:
                sub = re.split(r"(?m)^\s*(?=\d+\)\s*)", sec_zam)
                sub = [s.strip() for s in sub if s.strip()]
                if len(sub) > 1 and re.match(r"^\d+\)", sub[0]):
                    blocks = [("", s) for s in sub]
                else:
                    blocks = [("", sec_zam)]

        if not blocks:
            return [ZamawiajacyItem()]

        is_multi = len(blocks) > 1
        items: list[ZamawiajacyItem] = []
        for header_line, block in blocks:
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            first_line = re.sub(r"^\d+\)\s*", "", lines[0]) if lines else None
            nazwa = self._clean_entity_name(first_line)

            prowadzacy: int | None = None
            if is_multi:
                combined = f"{header_line}\n{block}"
                prowadzacy = 1 if re.search(r"lider|prowadząc\w*\s+postępowani", combined, re.I) else 0

            items.append(
                ZamawiajacyItem(
                    nazwa=nazwa,
                    krs=clean_digits(find_first(RE_KRS, block, 1)),
                    nip=clean_digits(find_first(RE_NIP, block, 1)),
                    regon=clean_digits(find_first(RE_REGON, block, 1)),
                    prowadzacy_postepowanie=prowadzacy,
                )
            )
        return items

    def _extract_zamowienie(self, text_clean: str) -> ZamowieniePubliczneData:
        sec_dotyczy = extract_section(
            text_clean,
            r"dotyczy\s*:",
            [r"Działając\s+w\s+imieniu", r"Zarzut\s+nr\s*1"],
        )
        scope = re.sub(r"\s+", " ", sec_dotyczy or text_clean[:2500])

        # BZP (bez spacji, nigdy TED)
        bzp_raw = find_first(RE_BZP, text_clean)
        numer_bzp = re.sub(r"\s+", "", bzp_raw).upper() if bzp_raw else None

        # TED (zawsze NNNNNN-YYYY)
        numer_ted = None
        ted_std = re.search(r"\b(\d{6}-\d{4})\b", text_clean)
        if ted_std:
            numer_ted = ted_std.group(1)
        else:
            ted_oj = re.search(r"\b(\d{4})/S\s*\d{3}-(\d{6})\b", text_clean, re.I)
            if ted_oj:
                numer_ted = f"{ted_oj.group(2)}-{ted_oj.group(1)}"

        # Numer postępowania
        proc_raw = find_first(_RE_PROC_NUM, scope, 1) or find_first(_RE_PROC_NUM, text_clean, 1)
        numer_postepowania = proc_raw.rstrip(".,;:") if proc_raw else None

        # Tytuł zamówienia
        tytul = None
        m_title = re.search(
            r'\bpn\.?\s*[:.]?\s*["„]([^"„”]+)["”]',
            scope,
            re.I,
        )
        if not m_title:
            m_title = re.search(
                r"\bpn\.?\s*[:.]?\s*(.+?)(?=\s*,?\s*(?:nr\s+referencyjny|numer\s+referencyjny|nr\s+sprawy|nr\s+postępowania|numer\s+postępowania|znak\s+sprawy|ogłoszenie\s+o\s+zamówieniu|numer\s+ogłoszenia))",
                scope,
                re.I,
            )
        if not m_title and sec_dotyczy:
            m_title = re.search(r'["„]([^"„”]{10,})["”]', scope)
        if m_title:
            tytul = " ".join(m_title.group(1).split()).strip(' "„”.,;:')

        return ZamowieniePubliczneData(
            numer_bzp=numer_bzp,
            numer_postepowania=numer_postepowania,
            numer_ted=numer_ted,
            tytul=tytul,
        )

    def extract(
        self,
        text_clean: str,
        entities: dict[str, list[str]] | None = None,
    ) -> OdwolanieSchema:
        miejscowosc, data_odwolania = self._extract_header_city_date(text_clean)
        czynnosc_dzien, czynnosc_powod_id, czynnosc_opis = self._extract_action(text_clean)

        return OdwolanieSchema(
            OdwolanieKIO=OdwolanieKIOData(
                data=data_odwolania,
                czynnosc_dzien=czynnosc_dzien,
                czynnosc_powod_id=czynnosc_powod_id,
                czynnosc_opis=czynnosc_opis,
                miejscowosc=miejscowosc,
            ),
            Odwolujacy=self._extract_odwolujacy(text_clean),
            Zamawiajacy=self._extract_zamawiajacy(text_clean),
            ZamowieniePubliczne=self._extract_zamowienie(text_clean),
        )