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
        r"(?:nr|numer)\s*(?:postępowania|referencyjny|sprawy)[:\s]*([A-Za-z0-9./_-]+)",
        re.IGNORECASE,
    )

    def extract(
        self,
        text_clean: str,
        entities: dict[str, list[str]] | None = None,
    ) -> OdwolanieSchema:
        city_match = self._RE_CITY_DATE.search(text_clean[:500])
        miejscowosc = city_match.group(1) if city_match else None
        data_odwolania = city_match.group(2) if city_match else find_first(RE_DATE, text_clean[:500], group=1)

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

        return OdwolanieSchema(
            OdwolanieKIO=OdwolanieKIOData(
                data=data_odwolania,
                miejscowosc=miejscowosc,
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
                    ),
                )
            ],
            Zamawiajacy=[
                ZamawiajacyItem(
                    nazwa=zam_lines[0] if zam_lines else None,
                    krs=clean_digits(find_first(RE_KRS, sec_zam, group=1)),
                    nip=clean_digits(find_first(RE_NIP, sec_zam, group=1)),
                    regon=clean_digits(find_first(RE_REGON, sec_zam, group=1)),
                )
            ],
            ZamowieniePubliczne=ZamowieniePubliczneData(
                numer_bzp=find_first(RE_BZP, text_clean),
                numer_ted=find_first(RE_TED, text_clean),
                numer_postepowania=find_first(self._RE_PROC_NUM, text_clean, group=1),
            ),
        )