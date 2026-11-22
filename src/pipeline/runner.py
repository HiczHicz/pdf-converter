import logging
from pathlib import Path
import spacy

from extraction import ExtractorRegistry
from pipeline.classifier import DocumentClassifier
from pipeline.io import ResultWriter, discover_pdfs
from pipeline.schemas import ProcessingResult
from preprocessing.factory import PipelineFactory
from utils import DataExplorer, extract_entities

logging.getLogger("pdfminer").setLevel(logging.ERROR)


class DocumentPipeline:
    def __init__(
        self,
        model_path: str | Path,
        explorer: DataExplorer | None = None,
        classifier: DocumentClassifier | None = None,
        spacy_model: str = "pl_core_news_sm",
        min_confidence: float = 0.0,
    ) -> None:
        self._explorer = explorer or DataExplorer(data_path=None)
        self._classifier = classifier or DocumentClassifier(model_path)
        self._min_confidence = min_confidence

        nlp = spacy.load(spacy_model, disable=["ner", "parser"])
        self._clean_pipe = PipelineFactory.build_for_extraction()
        self._nlp_pipe = PipelineFactory.build_for_tfidf(nlp)

    def process_file(
        self,
        pdf_path: Path,
        relative_path: Path,
        writer: ResultWriter | None = None,
    ) -> ProcessingResult:
        try:
            raw_text = self._explorer.read_pdf(pdf_path)
            if not raw_text.strip():
                output_json = writer.write_document(relative_path, {}) if writer else None
                return ProcessingResult(
                    source_pdf=pdf_path,
                    relative_path=relative_path,
                    label="negatywne",
                    confidence=0.0,
                    output_json=output_json,
                    data={},
                )
            text_clean = self._clean_pipe.run(raw_text)
            entities = extract_entities(text_clean)
            text_nlp = self._nlp_pipe.run(raw_text)

            label, confidence = self._classifier.predict(text_nlp)
            if confidence is not None and confidence < self._min_confidence:
                label = "negatywne"

            extractor = ExtractorRegistry.get(label)
            data = extractor.extract_dict(text_clean, entities) if extractor else {}
            output_json = writer.write_document(relative_path, data) if writer else None

            return ProcessingResult(
                source_pdf=pdf_path,
                relative_path=relative_path,
                label=label,
                confidence=confidence,
                output_json=output_json,
                data=data,
            )
        except Exception as exc:
                    output_json = writer.write_document(relative_path, {}) if writer else None
                    return ProcessingResult(
                        source_pdf=pdf_path,
                        relative_path=relative_path,
                        output_json=output_json,
                        data={},
                        error=str(exc),
                    )

    def run(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        verbose: bool = True,
    ) -> list[ProcessingResult]:
        base_dir, pdfs = discover_pdfs(Path(input_path))
        writer = ResultWriter(output_dir)
        results: list[ProcessingResult] = []

        for idx, pdf_path in enumerate(pdfs, start=1):
            rel_path = pdf_path.relative_to(base_dir)
            result = self.process_file(pdf_path, rel_path, writer=writer)
            results.append(result)

            if verbose:
                if result.error:
                    print(f"[{idx}/{len(pdfs)}] BŁĄD  {rel_path}: {result.error}", flush=True)
                else:
                    conf = f" ({result.confidence:.2%})" if result.confidence is not None else ""
                    status = f"-> {result.output_json.name}" if result.output_json else "(pominięto)"
                    print(f"[{idx}/{len(pdfs)}] {result.label}{conf} | {rel_path} {status}", flush=True)

        if verbose:
            print(f"\nGotowe. Przetworzono plików: {len(results)}", flush=True)
        return results