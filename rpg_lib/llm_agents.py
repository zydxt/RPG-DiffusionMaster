from typing import Tuple
from pathlib import Path
from modules import scripts
from openai import AzureOpenAI, OpenAI
from rpg_lib.rpg_enums import LLMType, PromptVersion, Quantization
from rpg_lib.logs import logger
import google.generativeai as genai
from transformers import AutoModelForCausalLM, AutoTokenizer

import re
import gc

PROMPT_TEMPLATE_FOLDER = Path(scripts.basedir()) / "prompt_template"
PROMPT_TEMPLATE_FILE = PROMPT_TEMPLATE_FOLDER / "template.txt"
MULTI_ATTRIBUTE_TEMPLATE_FILE = (
    PROMPT_TEMPLATE_FOLDER / "human_multi_attribute_examples.txt"
)
COMPLEX_OBJECT_TEMPLATE_FILE = (
    PROMPT_TEMPLATE_FOLDER / "complex_multi_object_examples.txt"
)
LOCAL_LLM_MODELS_FOLDER = Path(scripts.basedir()) / "models"


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
        return self.client.generate_content(text_prompt).text


class LocalLLM(LLMAgent):
    def __init__(
        self,
        model_id=None,
        model_folder=None,
        quantization=None,
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.model_folder = model_folder
        self.quantization_config = {
            "load_in_4bit": quantization == Quantization.Q_4_BIT,
            "load_in_8bit": quantization == Quantization.Q_8_BIT,
        }

    def _get_regional_content_from_llm(self, text_prompt) -> str:
        model_id_or_folder = self.model_id or self.model_folder

        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_folder,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=LOCAL_LLM_MODELS_FOLDER,
            **self.quantization_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_folder,
            trust_remote_code=True,
            cache_dir=LOCAL_LLM_MODELS_FOLDER,
        )

        inputs = tokenizer(
            text_prompt, return_tensors="pt", return_attention_mask=False
        )
        outputs = model.generate(**inputs, max_length=1024)

        result = tokenizer.batch_decode(outputs)[0]

        try:
            import torch

            del tokenizer
            del model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            self._log_text("LOCAL_LLM", "free up vram failed")

        return result


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

    if llm_type == LLMType.LOCAL_LLM:
        return LocalLLM(
            model_id=kwargs["model_id"],
            model_folder=kwargs["model_folder"],
            quantization=kwargs["quantization"],
        )

    return DummyAgent()
