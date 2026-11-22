from pydantic import BaseModel, ConfigDict, Field


# --- PRZELEW ---
class PrzelewBankowy(BaseModel):
    data_operacji: str | None = None
    kwota: float | str | None = None
    na_rachunek: str | None = None
    numer_transakcji: str | None = None
    tytul: str | None = None
    z_rachunku: str | None = None


class PrzelewSchema(BaseModel):
    przelew_bankowy: PrzelewBankowy = Field(default_factory=PrzelewBankowy)


# --- PEŁNOMOCNICTWO ---
class MocodawcaData(BaseModel):
    imie_nazwisko: str | None = None
    krs: str | None = None
    nazwa: str | None = None
    nip: str | None = None
    typ: str | None = None


class PodpisujacyItem(BaseModel):
    imie_nazwisko: str | None = None


class MocodawcaItem(BaseModel):
    data: MocodawcaData = Field(default_factory=MocodawcaData)
    podpisujacy: list[PodpisujacyItem] = Field(default_factory=lambda: [PodpisujacyItem()])


class PelnomocnikItem(BaseModel):
    imie_nazwisko: str | None = None
    pesel: str | None = None


class PelnomocnictwoSchema(BaseModel):
    mocodawca: list[MocodawcaItem] = Field(default_factory=lambda: [MocodawcaItem()])
    pelnomocnicy: list[PelnomocnikItem] = Field(default_factory=lambda: [PelnomocnikItem()])


# --- ODWOŁANIE ---
class OdwolanieKIOData(BaseModel):
    data: str | None = None
    czynnosc_dzien: str | None = None
    czynnosc_powod_id: int | None = None
    czynnosc_opis: str | None = None
    miejscowosc: str | None = None


class OdwolujacyPelnomocnik(BaseModel):
    imie_nazwisko: str | None = None
    nazwa: str | None = None
    mail: str | None = None


class OdwolujacyItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    krs: str | None = None
    nazwa: str | None = None
    nip: str | None = None
    regon: str | None = None
    przez_pelnomocnika: int | None = Field(default=None, alias="przez-pelnomocnika")
    pelnomocnik: OdwolujacyPelnomocnik = Field(default_factory=OdwolujacyPelnomocnik)


class ZamawiajacyItem(BaseModel):
    krs: str | None = None
    nazwa: str | None = None
    nip: str | None = None
    regon: str | None = None
    prowadzacy_postepowanie: int | None = None


class ZamowieniePubliczneData(BaseModel):
    numer_bzp: str | None = None
    numer_postepowania: str | None = None
    numer_ted: str | None = None
    tytul: str | None = None


class OdwolanieSchema(BaseModel):
    OdwolanieKIO: OdwolanieKIOData = Field(default_factory=OdwolanieKIOData)
    Odwolujacy: list[OdwolujacyItem] = Field(default_factory=lambda: [OdwolujacyItem()])
    Zamawiajacy: list[ZamawiajacyItem] = Field(default_factory=lambda: [ZamawiajacyItem()])
    ZamowieniePubliczne: ZamowieniePubliczneData = Field(default_factory=ZamowieniePubliczneData)