from gradio.components.base import Component
from rpg_lib.llm_agents import llm_factory
from rpg_lib.rpg_enums import PromptVersion, LLMType
from rpg_lib.logs import change_debug
import modules.scripts as scripts
import gradio as gr


class RPGDiffusionMasterScript(scripts.Script):
    def __init__(self) -> None:
        super().__init__()
        self.rp_activate = None
        self.rp_main = None
        self.rp_mode_tab = None
        self.rp_main_splitting = None
        self.rp_generation_mode = None
        self.rp_divide_ratio = None
        self.rp_base_ratio = None
        self.rp_usebase = None
        self.rp_base_ratio = None
        self.txt2img_prompt = None

    def title(self):
        return "RPG DiffusionMaster"

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, is_img2img):
        with gr.Accordion("RPG DiffusionMaster", elem_id="RPG_DM_main"):
            with gr.Column():
                prompt_version = gr.Dropdown(
                    label="Prompt Version",
                    choices=[m.value for m in PromptVersion],
                    elem_id="RPG_DM_prompt_version",
                )
                base_prompt = gr.Textbox(
                    label="Base Prompt",
                    lines=2,
                    placeholder="(Optional) Input your base prompt. Leave it empty if you don't want a base prompt",
                    elem_id="RPG_DM_base_prompt",
                )
                base_ratio = gr.Textbox(
                    label="Base Ratio",
                    lines=1,
                    max_lines=1,
                    value="0.3",
                    elem_id="RPG_DM_base_ratio",
                )
                user_prompt = gr.Textbox(
                    lines=3,
                    placeholder="Input prompt here",
                    show_label=False,
                    elem_id="RPG_DM_user_prompt",
                )

            with gr.Column():
                gr.Markdown("LLM config")
                llm_model = gr.Dropdown(
                    label="Model",
                    choices=[m.value for m in LLMType],
                    elem_id="RPG_DM_model",
                )
                api_key = gr.Textbox(
                    label="API Key", lines=1, max_lines=1, elem_id="RPG_DM_api_key", visible=False,
                )
                model_path = gr.Textbox(
                    label="Model Path", 
                    lines=1, 
                    max_lines=1, 
                    visible=False,
                    elem_id="RPG_Model",
                )
                gpu_layers = gr.Textbox(
                    label="GPU Layers",
                    lines=1,
                    max_lines=1,
                    visible=False,
                    elem_id="RPG_GPU_layers",
                )

                azure_endpoint = gr.Textbox(
                    label="Azure endpoint",
                    lines=1,
                    placeholder="https://YOUR_RESOURCE_NAME.openai.azure.com",
                    visible=False,
                    elem_id="RPG_DM_azure_endpoint",
                )
                azure_api_version = gr.Textbox(
                    label="Azure api version",
                    lines=1,
                    placeholder="2023-07-01-preview",
                    visible=False,
                    elem_id="RPG_DM_azure_api_version",
                )
                azure_deployment_name = gr.Textbox(
                    label="Azure deployment name",
                    lines=1,
                    placeholder="deployment name",
                    visible=False,
                    elem_id="RPG_DM_azure_deployment_name",
                )

            debug = gr.Checkbox(label="Debug", elem_id="RPG_DM_debug")
            apply = gr.Button("Apply to Prompt", elem_id="RPG_DM_apply")

        def select_model(model):
            enable_azure_config = model == LLMType.GPT4_AZURE.value
            enable_local_model = model == LLMType.Local.value
            
            local_update = gr.Textbox.update(
                visible=enable_local_model,
                interactive=enable_local_model,
                value="",
            )

            api_update = gr.Textbox.update(
                visible=not enable_local_model,
                interactive=not enable_local_model,
                value="",
            )

            gpu_update = gr.Textbox.update(
                visible=enable_local_model,
                interactive=enable_local_model,
                value="0",
            )

            update = gr.Textbox.update(
                visible=enable_azure_config,
                interactive=enable_azure_config,
                value="",
            )

            return {
                gpu_layers: gpu_update,
                model_path: local_update,
                api_key: api_update,
                azure_endpoint: update,
                azure_api_version: update,
                azure_deployment_name: update,
            }

        llm_model.change(
            fn=select_model,
            inputs=[llm_model],
            outputs=[gpu_layers, model_path, api_key, azure_endpoint, azure_api_version, azure_deployment_name],
        )

        apply.click(
            fn=self.apply_change,
            inputs=[
                prompt_version,
                user_prompt,
                base_prompt,
                base_ratio,
                llm_model,
                gpu_layers,
                model_path,
                api_key,
                azure_endpoint,
                azure_api_version,
                azure_deployment_name,
            ],
            outputs=[
                c
                for c in (
                    self.rp_main,
                    self.rp_activate,
                    self.rp_mode_tab,
                    self.rp_main_splitting,
                    self.rp_generation_mode,
                    self.rp_divide_ratio,
                    self.rp_usebase,
                    self.rp_base_ratio,
                    self.txt2img_prompt,
                )
                if c
            ],
        )

        debug.change(fn=change_debug, inputs=debug, outputs=None)

        return [
            prompt_version,
            user_prompt,
            base_prompt,
            base_ratio,
            llm_model,
            api_key,
            azure_endpoint,
            azure_api_version,
            azure_deployment_name,
            debug,
        ]

    def apply_change(
        self,
        prompt_version: str,
        user_prompt: str,
        base_prompt: str,
        base_ratio: str,
        llm_model: str,
        gpu_layers: str,
        model_path: str,
        api_key: str,
        azure_endpoint: str,
        azure_api_version: str,
        azure_deployment_name: str,
    ):
        split_ratio, regional_prompt = llm_factory(
            LLMType(llm_model),
            gpu_layers=gpu_layers,
            model_path=model_path,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_api_version,
            azure_deployment_name=azure_deployment_name,
        ).get_regional_parameter(user_prompt, PromptVersion(prompt_version))

        if _base := (base_prompt or base_prompt.strip()):
            final_prompt = " BREAK\n".join([_base, regional_prompt])
            use_base = True
        else:
            final_prompt = " BREAK\n".join([user_prompt, regional_prompt])
            use_base = False

        return {
            self.rp_main: gr.Accordion.update(open=True),
            self.rp_activate: gr.Checkbox.update(value=True),
            self.rp_mode_tab: gr.Tabs.update(selected="tMatrix"),
            self.rp_main_splitting: gr.Radio.update(value="Columns"),
            self.rp_generation_mode: gr.Radio.update(value="Attention"),
            self.rp_divide_ratio: gr.Radio.update(value=f"{split_ratio}"),
            self.rp_usebase: gr.Checkbox.update(value=use_base),
            self.rp_base_ratio: gr.Textbox.update(value=base_ratio),
            self.txt2img_prompt: gr.Textbox.update(value=final_prompt),
        }

    def after_component(self, component: Component, **kwargs):
        if elem_id := kwargs.get("elem_id"):
            if elem_id == "RP_main":
                self.rp_main = component
            elif elem_id == "RP_active":
                self.rp_activate = component
            elif elem_id == "RP_mode":
                self.rp_mode_tab = component
            elif elem_id == "RP_main_splitting":
                self.rp_main_splitting = component
            elif elem_id == "RP_generation_mode":
                self.rp_generation_mode = component
            elif elem_id == "RP_divide_ratio":
                self.rp_divide_ratio = component
            elif elem_id == "RP_base_ratio":
                self.rp_base_ratio = component
            elif elem_id == "RP_usebase":
                self.rp_usebase = component
            elif elem_id == "RP_base_ratio":
                self.rp_base_ratio = component
            elif elem_id == "txt2img_prompt":
                self.txt2img_prompt = component
