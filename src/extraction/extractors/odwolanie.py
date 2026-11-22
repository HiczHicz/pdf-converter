import re

from extraction.base import BaseExtractor, ExtractorRegistry
from extraction.helpers import (
    RE_BZP,
    RE_DATE,
    RE_EMAIL,
    RE_KRS,
    RE_NIP,
    RE_REGON,
    RE_TED,
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


@ExtractorRegistry.register("odwolanie")
class OdwolanieExtractor(BaseExtractor):
    _RE_CITY_DATE = re.compile(
        r"([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż-]+),?\s*(?:dnia\s*)?(\d{4}-\d{2}-\d{2}|\d{2}[./-]\d{2}[./-]\d{4})"
    )
    _RE_PROC_NUM = re.compile(
        r"(?:nr|numer)\s*(?:referencyjny\s+sprawy|referencyjny|postępowania|sprawy)\s*[:]\s*([A-Za-z0-9./_-]+)",
        re.IGNORECASE,
    )
    _RE_PERSON = re.compile(
        r"(?:r\.\s*pr\.|adw\.|radca\s+prawny|adwokat)\s*"
        r"([A-ZĄĆĘŁŃÓŚŹŻ][A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż-]+\s+"
        r"[A-ZĄĆĘŁŃÓŚŹŻ][A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż-]+)",
        re.IGNORECASE,
    )
    _RE_ACTIONS = (
        (re.compile(r"zaniechał\s+unieważnienia\s+postępowania", re.I), "Zaniechanie unieważnienia postępowania", 10),
        (re.compile(r"unieważn.{0,12}(?:postępowanie|postępowania|rozstrzygnięcie|rozstrzygnięcia)", re.I), "Decyzja o unieważnieniu postępowania", 9),
        (re.compile(r"zaniechał\s+odrzucenia\s+oferty|nie\s+odrzucił\s+oferty", re.I), "Zaniechanie odrzucenia oferty", 7),
        (re.compile(r"zaniechał\s+wykluczenia|nie\s+wykluczył", re.I), "Zaniechanie wykluczenia wykonawcy", 8),
        (re.compile(r"Zamawiający[^\n]{0,180}(?:unieważni[^\s]*\s+postępowania|poinformował\s+o\s+unieważnieniu)", re.I), "Decyzja o unieważnieniu postępowania", 9),
        (re.compile(r"Zamawiający[^\n]{0,180}poinformował\s+o\s+odrzuceniu\s+oferty|Zamawiający[^\n]{0,180}odrzucił\s+ofertę", re.I), "Decyzja o odrzuceniu oferty", 1),
        (re.compile(r"Zamawiający[^\n]{0,180}ogłosił\s+wybór\s+oferty", re.I), "Ogłoszenie wyboru oferty", 2),
        (re.compile(r"Zamawiający[^\n]{0,180}(?:poinformował\s+o\s+)?wykluczył|Zamawiający[^\n]{0,180}wykluczeniu\s+z", re.I), "Decyzja o wykluczeniu wykonawcy", 3),
        (re.compile(r"opublikował\s+(?:Specyfikację|Specyfikacji)\s+Warunków\s+Zamówienia", re.I), "Publikacja specyfikacji warunków zamówienia (SWZ)", 4),
        (re.compile(r"zmodyfikował\s+treść\s+(?:Specyfikacji|SWZ)|modyfikac(?:ja|ji)\s+treści\s+SWZ", re.I), "Modyfikacja treści SWZ", 5),
        (re.compile(r"przekazał[^\n]*(?:odpowiedzi|wyjaśnienia)|(?:odpowiedzi|wyjaśnienia)[^\n]*Specyfikacji", re.I), "Odpowiedzi/wyjaśnienia do SWZ", 6),
        (re.compile(r"Zamawiający[^\n]{0,180}(?:unieważni(?:ł|ił|iił)|poinformował\s+o\s+unieważnieniu)\s+(?:postępowania|rozstrzygnięcia)", re.I), "Decyzja o unieważnieniu postępowania", 9),
        (re.compile(r"udostępnił\s+Opis\s+Przedmiotu\s+Zamówienia|OPZ", re.I), "Opis przedmiotu zamówienia (OPZ)", 11),
        (re.compile(r"projektowan(?:e|ych)\s+postanowie(?:ń|nia)|postanowień\s+umowy", re.I), "Projektowane postanowienia umowy", 12),
    )

    def extract(
        self,
        text_clean: str,
        entities: dict[str, list[str]] | None = None,
    ) -> OdwolanieSchema:
        signature_match = re.search(
            r"([A-ZĄĆĘŁŃÓŚŹŻ][A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż-]+)[, ]+\s*(?:dnia\s+)?"
            r"(\d{1,2}\s+[A-Za-ząćęłńóśźż]+\s+\d{4})",
            text_clean,
        )
        if signature_match:
            miejscowosc = signature_match.group(1)
            data_odwolania = normalize_polish_date(signature_match.group(2))
        else:
            numeric_signature = re.search(
                r"([A-ZĄĆĘŁŃÓŚŹŻ][A-Za-zĄĆĘŁŃÓŚŹŻ-]+),?\s+(?:dnia\s+)?"
                r"(\d{1,2}[./-]\d{1,2}[./-]\d{4})",
                text_clean,
            )
            if numeric_signature:
                miejscowosc = numeric_signature.group(1)
                data_odwolania = normalize_polish_date(numeric_signature.group(2))
            else:
                city_match = self._RE_CITY_DATE.search(text_clean[:500])
                miejscowosc = city_match.group(1) if city_match else None
                data_odwolania = normalize_polish_date(city_match.group(2) if city_match else find_first(RE_DATE, text_clean[:500], group=1))

        sec_odw = extract_section(
            text_clean,
            r"odwołujący[:\s]*",
            [r"zamawiający[:\s]*", r"dotyczy[:\s]*", r"odwołanie\b"],
        )
        sec_zam = extract_section(
            text_clean,
            r"zamawiający[:\s]*",
            [r"dotyczy[:\s]*", r"postępowanie[:\s]*", r"odwołanie\b", r"przedmiot\b"],
        )

        odw_lines = [ln.strip() for ln in sec_odw.splitlines() if ln.strip()]
        zam_lines = [ln.strip() for ln in sec_zam.splitlines() if ln.strip()]

        has_proxy = bool(re.search(r"reprezentowan|pełnomocni", sec_odw, re.IGNORECASE))
        proxy_match = find_first(self._RE_PERSON, sec_odw, group=1)

        proxy_name = None
        proxy_text = re.sub(r"\s+", " ", sec_odw)
        proxy_name_match = re.search(
            r"(Kancelaria[^,]+(?:odpowiedzialnością)?)(?=,\s*ul\.)",
            proxy_text,
            re.IGNORECASE,
        )
        if proxy_name_match:
            proxy_name = " ".join(proxy_name_match.group(1).split())

        action_text = text_clean
        action_dzien = None
        action_match = re.search(
            r"Zamawiający\b[^\n]{0,160}?w\s+dniu\s+(\d{1,2}\s+[A-Za-ząćęłńóśźż]+\s+\d{4})",
            action_text,
            re.IGNORECASE,
        )
        if action_match:
            action_dzien = normalize_polish_date(action_match.group(1))
        if not action_dzien:
            action_match = re.search(
                r"(?:w\s+dniu|dnia)\s+(\d{1,2}\s+[A-Za-ząćęłńóśźż]+\s+\d{4})",
                action_text,
                re.IGNORECASE,
            )
            action_dzien = normalize_polish_date(action_match.group(1)) if action_match else None

        czynnosc_opis = None
        czynnosc_powod_id = None
        for pattern, description, cause_id in self._RE_ACTIONS:
            if pattern.search(action_text):
                czynnosc_opis = description
                czynnosc_powod_id = cause_id
                break

        zam_items: list[ZamawiajacyItem] = []
        leader_match = re.search(
            r"(?:1\)\s*)?Zamawiający\s*-\s*Lider(?:,\s*prowadzący postępowanie)?\s*:\s*"
            r"([^\n]+)",
            text_clean,
            re.IGNORECASE,
        )
        partner_match = re.search(
            r"(?:2\)\s*)?Zamawiający\s*-\s*Partner\s*:\s*([^\n]+)",
            text_clean,
            re.IGNORECASE,
        )
        if leader_match:
            leader_block = text_clean[
                leader_match.start():partner_match.start() if partner_match else len(text_clean)
            ]
            leader_name = " ".join(leader_match.group(1).split())
            zam_items.append(ZamawiajacyItem(
                nazwa=leader_name,
                krs=clean_digits(find_first(RE_KRS, leader_block, group=1)),
                nip=clean_digits(find_first(RE_NIP, leader_block, group=1)),
                regon=clean_digits(find_first(RE_REGON, leader_block, group=1)),
                prowadzacy_postepowanie=1,
            ))
        if partner_match:
            partner_block = text_clean[partner_match.start():]
            partner_name = " ".join(partner_match.group(1).split())
            partner_end = re.search(r"\n(?:dotyczy|odwołujący|postępowanie)\s*:", partner_block, re.I)
            if partner_end:
                partner_block = partner_block[:partner_end.start()]
            zam_items.append(ZamawiajacyItem(
                nazwa=partner_name,
                krs=clean_digits(find_first(RE_KRS, partner_block, group=1)),
                nip=clean_digits(find_first(RE_NIP, partner_block, group=1)),
                regon=clean_digits(find_first(RE_REGON, partner_block, group=1)),
                prowadzacy_postepowanie=0,
            ))
        if not zam_items:
            zam_items = [ZamawiajacyItem(
                nazwa=zam_lines[0] if zam_lines else None,
                krs=clean_digits(find_first(RE_KRS, sec_zam, group=1)),
                nip=clean_digits(find_first(RE_NIP, sec_zam, group=1)),
                regon=clean_digits(find_first(RE_REGON, sec_zam, group=1)),
            )]

        proc_match = re.search(
            r"(?:nr\s+referencyjny\s+sprawy|nr\s+sprawy|numer\s+postępowania|nr\s+postępowania)"
            r"\s*[:]\s*([A-Za-z0-9./_-]+)",
            text_clean,
            re.IGNORECASE,
        )
        proc_num = proc_match.group(1) if proc_match else find_first(self._RE_PROC_NUM, text_clean, group=1)

        title_match = re.search(
            r"\bpn\.?\s*(?:[:.]?\s*)(?:[„\"]([^„”\"]+)[”\"]|([^\n]+?))"
            r"(?=\s*(?:nr\s+(?:referencyjny\s+sprawy|sprawy|postępowania)\s*:|,?\s*ogłoszenie))",
            text_clean,
            re.IGNORECASE,
        )
        title = None
        if title_match:
            title = " ".join((title_match.group(1) or title_match.group(2) or "").split()).strip(" \"„”")
        if not title:
            title_match = re.search(r'[„"]([^„”"]+)[”"]', text_clean)
            title = " ".join(title_match.group(1).split()) if title_match else None

        ted_match = re.search(r"\b\d{6}-\d{4}\b", text_clean)
        if not ted_match:
            ted_match = re.search(r"\b\d{4}/S\s*\d{3}-\d{6}\b", text_clean, re.I)
            if ted_match:
                suffix = re.search(r"S\s*\d{3}-(\d{6})", ted_match.group(0), re.I)
                ted_value = f"{suffix.group(1)}-{ted_match.group(0)[:4]}" if suffix else ted_match.group(0)
            else:
                ted_value = None
        else:
            ted_value = ted_match.group(0)

        return OdwolanieSchema(
            OdwolanieKIO=OdwolanieKIOData(
                data=data_odwolania,
                miejscowosc=miejscowosc,
                czynnosc_dzien=action_dzien,
                czynnosc_opis=czynnosc_opis,
                czynnosc_powod_id=czynnosc_powod_id,
            ),
            Odwolujacy=[
                OdwolujacyItem(
                    nazwa=odw_lines[0] if odw_lines else None,
                    krs=clean_digits(find_first(RE_KRS, sec_odw, group=1)),
                    nip=clean_digits(find_first(RE_NIP, sec_odw, group=1)),
                    regon=clean_digits(find_first(RE_REGON, sec_odw, group=1)),
                    przez_pelnomocnika=has_proxy,
                    pelnomocnik=OdwolujacyPelnomocnik(
                        mail=find_first(RE_EMAIL, sec_odw),
                        imie_nazwisko=proxy_match,
                        nazwa=proxy_name,
                    ),
                )
            ],
            Zamawiajacy=zam_items,
            ZamowieniePubliczne=ZamowieniePubliczneData(
                numer_bzp=find_first(RE_BZP, text_clean),
                numer_ted=ted_value,
                numer_postepowania=proc_num,
                tytul=title,
            ),
        )