from typing import Tuple
from pathlib import Path
from modules import scripts
from openai import AzureOpenAI, OpenAI
from rpg_lib.rpg_enums import LLMType, PromptVersion
from rpg_lib.logs import logger
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai
import re
from llama_cpp import Llama
from .openrouter_api import OpenRouterAPI

PROMPT_TEMPLATE_FOLDER = Path(scripts.basedir()) / "prompt_template"
PROMPT_TEMPLATE_FILE = PROMPT_TEMPLATE_FOLDER / "template.txt"
MULTI_ATTRIBUTE_TEMPLATE_FILE = (
    PROMPT_TEMPLATE_FOLDER / "human_multi_attribute_examples.txt"
)
COMPLEX_OBJECT_TEMPLATE_FILE = (
    PROMPT_TEMPLATE_FOLDER / "complex_multi_object_examples.txt"
)


class LLMAgent:
    @staticmethod
    def _log_text(name, text, log_fn=logger.debug):
        log_fn(f"[{name}]\n{text}\n[LOG_END]")

    def get_regional_parameter(self, prompt, version: PromptVersion) -> Tuple[str, str]:
        text_prompt = self._construct_llm_prompt(prompt, version)
        content = self._get_regional_content_from_llm(text_prompt)
        self._log_text("PLAN_CONTENT_FROM_LLM", content)
        return self._get_params_dict(content)

    def _construct_llm_prompt(self, prompt, version) -> str:
        template = PROMPT_TEMPLATE_FILE.read_text()
        if version == PromptVersion.MULTI_ATTRIBUTE:
            incontext_examples = MULTI_ATTRIBUTE_TEMPLATE_FILE.read_text()
        elif version == PromptVersion.COMPLEX_OBJECT:
            incontext_examples = COMPLEX_OBJECT_TEMPLATE_FILE.read_text()
        else:
            raise ValueError("PromptVersion Error")
        user_textprompt = f"Caption: {prompt} \n Let's think step by step:"
        llm_prompt = f"{template} \n {incontext_examples} \n{user_textprompt}"
        self._log_text("PROMPT_TO_LLM", llm_prompt)
        return llm_prompt

    def _get_regional_content_from_llm(self, text_prompt) -> str:
        # Implement this method to add new model
        raise NotImplementedError()

    def _get_params_dict(self, content):
        output_text = self._extract_output(content)
        self._log_text("EXTRACTED_OUTPUT_PART", output_text)

        output_text = self._extract_output(content)

        if mat := re.search(r"Split ratio: (.*)", output_text):
            split_ratio = mat.group(1).replace("(", "").replace(")", "").strip()
        else:
            self._log_text("EXTRACTED_OUTPUT_PART", output_text, logger.error)
            raise ValueError(f"Split ratio not found in output")

        if mat := re.search(r"Prompt:(.*)", output_text, re.DOTALL):
            regional_prompt = mat.group(1).strip()
        else:
            self._log_text("EXTRACTED_OUTPUT_PART", output_text, logger.error)
            raise ValueError(f"Ragional Prompt not found in output")

        self._log_text("RATIO_FROM_OUTPUT", split_ratio)
        self._log_text("REGIONAL_PROMPT_FROM_OUTPUT", regional_prompt)
        return split_ratio, regional_prompt

    def _extract_output(self, text):
        if mat := re.search(r"### Output:(.*?)(?=###|$)", text, re.DOTALL):
            return mat.group(1).strip()
        else:
            return ""


class DummyAgent(LLMAgent):
    # Just for demo
    def get_regional_parameter(self, prompt, version) -> Tuple[str, str]:
        return (
            "1;1;1",
            "Golden blonde hair illuminated in ambient light, styled in flowing waves that frame the girl's confident and engaging demeanor. BREAK\nSleek black suit tailored to perfection, expressing the sharpness of the lapels and the smoothness of the fabric, signifying sophistication. BREAK\nPristine white skirt, its hem dancing lightly above the knee, clean lines and simplicity outline its elegance, paired with the suit for a classic monochrome look.",
        )


class GPT4(LLMAgent):
    def __init__(self, api_key) -> None:
        self.client = OpenAI(
            api_key=api_key,
        )

    def _get_regional_content_from_llm(self, text_prompt) -> str:
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": text_prompt,
                },
            ],
            model="gpt-4-1106-preview",
        )
        return response.choices[0].message.content or ""


class GPT4Azure(LLMAgent):
    def __init__(
        self, api_key, azure_endpoint, azure_api_version, azure_deployment_name
    ) -> None:
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=azure_api_version,
        )
        self.deployment_name = azure_deployment_name

    def _get_regional_content_from_llm(self, text_prompt) -> str:
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {
                    "role": "user",
                    "content": text_prompt,
                },
            ],
            max_tokens=1024,  # fix completions stop for max-token issue when use gpt4-vision model
        )
        return response.choices[0].message.content or ""


class GeminiPro(LLMAgent):
    def __init__(self, api_key) -> None:
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model_name="gemini-pro")

    def _get_regional_content_from_llm(self, text_prompt) -> str:
        return self.client.generate_content(
            text_prompt,
            safety_settings={
                HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DEROGATORY: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_TOXICITY: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_VIOLENCE: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUAL: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_MEDICAL: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        ).text


class LocalModel(LLMAgent):
    def __init__(self, model_path, gpu_layers) -> None:
        self.client = Llama(
            model_path=model_path,
            chat_format="chatml",
            n_ctx=4096,
            n_gpu_layers=int(gpu_layers),
        )

    def _get_regional_content_from_llm(self, text_prompt) -> str:
        response = self.client.create_chat_completion_openai_v1(
            messages=[
                {
                    "role": "user",
                    "content": text_prompt,
                },
            ],
        )
        del self.client
        return response.choices[0].message.content or ""


class OpenRouterAgent(LLMAgent):
    def __init__(self, api_key: str, model_name: str) -> None:
        self.client = OpenRouterAPI(api_key)
        self.model_name = model_name

    def _get_regional_content_from_llm(self, text_prompt) -> str:
        return self.client.chat_completion(
            model=self.model_name, messages=[{"role": "user", "content": text_prompt}]
        )

    @staticmethod
    def get_available_models(api_key: str) -> list[str]:
        """Get list of available models from OpenRouter"""
        client = OpenRouterAPI(api_key)
        return client.get_available_models()


def llm_factory(
    llm_type,
    **kwargs,
) -> LLMAgent:
    if llm_type == LLMType.GPT4_AZURE:
        return GPT4Azure(
            kwargs["api_key"],
            kwargs["azure_endpoint"],
            kwargs["azure_api_version"],
            kwargs["azure_deployment_name"],
        )

    if llm_type == LLMType.GPT4:
        return GPT4(kwargs["api_key"])

    if llm_type == LLMType.GEMINI_PRO:
        return GeminiPro(kwargs["api_key"])

    if llm_type == LLMType.Local:
        return LocalModel(kwargs["model_path"], kwargs["gpu_layers"])

    if llm_type == LLMType.OPENROUTER:
        return OpenRouterAgent(kwargs["api_key"], kwargs["model_name"])

    return DummyAgent()
