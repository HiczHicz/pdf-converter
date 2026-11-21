from spacy.language import Language

from .base import TextPipeline, TextStep
from .cleaners import BoilerplateRemover, Dehyphenator, UnicodeNormalizer, WhitespaceCleaner
from .maskers import EntityMasker
from .nlp_steps import Lowercaser, SpacyLemmatizer
from .patterns import BOILERPLATE_PATTERNS, ENTITY_PATTERNS


class PipelineFactory:
    @staticmethod
    def _base_steps() -> list[TextStep]:
        return [
            UnicodeNormalizer(),
            Dehyphenator(),
            BoilerplateRemover(BOILERPLATE_PATTERNS),
            WhitespaceCleaner(),
        ]

    @classmethod
    def build_for_extraction(cls) -> TextPipeline:
        return TextPipeline(cls._base_steps())

    @classmethod
    def build_for_embeddings(cls) -> TextPipeline:
        steps = cls._base_steps() + [EntityMasker(ENTITY_PATTERNS)]
        return TextPipeline(steps)

    @classmethod
    def build_for_tfidf(cls, nlp: Language) -> TextPipeline:
        steps = cls._base_steps() + [
            EntityMasker(ENTITY_PATTERNS),
            Lowercaser(),
            SpacyLemmatizer(nlp),
        ]
        return TextPipeline(steps)