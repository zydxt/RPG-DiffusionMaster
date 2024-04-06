import launch
import os

if not launch.is_installed("openai"):
    launch.run_pip("install openai==1.10.0", "requirements RPG-DiffusionMaster")
if not launch.is_installed("google-generativeai"):
    launch.run_pip("install google-generativeai==0.3.2", "requirements RPG-DiffusionMaster")
if not launch.is_installed("llama-cpp-python"):
    import torch
    if torch.cuda.is_available:
        os.environ["CMAKE_ARGS"] = "-DLLAMA_CUBLAS=on"
        print("Installing llama-cpp with cuda")
    launch.run_pip("install llama-cpp-python==0.2.56", "requirements RPG-DiffusionMaster")