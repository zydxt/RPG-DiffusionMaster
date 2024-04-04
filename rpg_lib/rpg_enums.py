from enum import Enum


class PromptVersion(Enum):
    MULTI_ATTRIBUTE = "multi-attribute"
    COMPLEX_OBJECT = "complex-object"


class LLMType(Enum):
    GEMINI_PRO = "gemini-pro"
    GPT4 = "gpt4"
    GPT4_AZURE = "GPT4-AZURE"
    LOCAL_LLM = "Local-LLM"


class Quantization(Enum):
    Q_4_BIT = "4BIT"
    Q_8_BIT = "8BIT"
    NON_Q = "Non_quantization"
