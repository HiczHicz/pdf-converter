# PDFdestroyer

Autonomiczny pipeline CLI działający lokalnie (offline, wyłącznie na CPU) do przetwarzania dokumentacji w postępowaniach przed Krajową Izbą Odwoławczą (KIO). Wczytuje wsadowo pliki PDF, klasyfikuje je i zapisuje ustandaryzowane, zwalidowane pliki JSON.

## Funkcjonalności
- **Hybrydowy odczyt PDF + OCR:** ekstrakcja tekstu przez `pdfplumber` i `PyMuPDF` z automatycznym przełączeniem na `RapidOCR` dla skanów bez warstwy tekstowej.
- **Klasyfikacja NLP (4 klasy):** polski preprocessing (`spaCy` + maskowanie encji regexami) i klasyfikator `scikit-learn` (`odwolanie`, `pelnomocnictwo`, `przelew`, `negatywne`).
- **Deterministyczna ekstrakcja strukturalna:** modułowe ekstraktory z walidacją `Pydantic` (normalizacja dat ISO, kody przyczyn KIO 1–12, numery BZP/TED, struktury konsorcjów, wielokrotni zamawiający/podpisujący, dane przelewów).
- **Odporność na błędy:** automatyczne pomijanie plików innych niż `.pdf` oraz bezpieczny zapis `{}` dla dokumentów negatywnych i uszkodzonych.

## Wyniki klasyfikatora
- Zbiór treningowy: 237 dokumentów (5-fold CV): **F1 macro = 1.0000**
- Zbiór walidacyjny: 60 dokumentów: **F1 macro = 1.0000**

## Struktura projektu
```text
├── main.py                              # Punkt wejścia CLI
├── requirements.txt                     # Zależności (macOS / Linux)
├── RUN.md                               # Komenda uruchomienia na VM
├── src/
│   ├── utils.py                         # Odczyt PDF, OCR, ekstrakcja encji
│   ├── preprocessing/                   # Czyszczenie, maskowanie, lematyzacja
│   ├── pipeline/                        # Orkiestracja, klasyfikacja, I/O
│   ├── extraction/                      # Schematy Pydantic i ekstraktory domenowe
│   └── created_models/                  # Zapisany model .joblib
├── input/                               # Katalog wejściowy na pliki PDF
└── output/                              # Katalog wyjściowy na pliki JSON

```
## Instalacja i uruchomienie
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download pl_core_news_sm

python main.py ./input -o ./output -m ./src/created_models/simple_text_classifier.joblib