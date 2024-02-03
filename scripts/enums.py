from enum import StrEnum, auto


class PromptVersion(StrEnum):
    MULTI_ATTRIBUTE = "multi-attribute"
    COMPLEX_OBJECT = "complex-object"


class LLMType(StrEnum):
    GEMINI_PRO = auto()
    GPT4 = auto()
    GPT4_AZURE = auto()
