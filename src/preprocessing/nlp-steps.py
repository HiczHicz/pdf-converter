from spacy.language import Language


class Lowercaser:
    def process(self, text: str) -> str:
        return text.lower()


class SpacyLemmatizer:
    def __init__(self, nlp: Language, keep_tokens_prefix: str = "token_") -> None:
        self._nlp = nlp
        self._prefix = keep_tokens_prefix

    def process(self, text: str) -> str:
        doc = self._nlp(text)
        lemmas = [
            token.text if token.text.startswith(self._prefix) else token.lemma_
            for token in doc
            if not token.is_punct and not token.is_space and not token.is_stop
        ]
        return " ".join(lemmas)