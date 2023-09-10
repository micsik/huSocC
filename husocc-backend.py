from configparser import SectionProxy
import sys
from typing import Any
from husocc import HuSocClassifier

from annif.project import AnnifProject
from annif.backend import backend
from annif.suggestion import SubjectSuggestion
from annif.exception import NotSupportedException

class HuSocBackend(backend.AnnifBackend):

    is_trained = True

    name = "husocc"

    language = "hu"

    classifier = None

    vocab = None

    def __init__(self, backend_id: str, config_params: dict[str, Any] | SectionProxy, project: AnnifProject) -> None:
        super().__init__(backend_id, config_params, project)
        # logger.info("HuSocBackend INIT")
        self.classifier = HuSocClassifier()
        self.vocab = project.vocab.subjects()

    def initialize(self, parallel: bool = False) -> None:
        # logger.info("HuSocBackend initialize")
        return super().initialize(parallel)

    def _suggest(self, text, params):
        topicList = self.classifier.findTopics(text)
        suggestions = [SubjectSuggestion(self.vocab.by_label(label, self.language), score) for (label, score) in topicList]
        return suggestions

    def _train(self, corpus, params, jobs=0):
        raise NotSupportedException("Training is not possible")
