import launch

if not launch.is_installed("openai"):
    launch.run_pip("install openai==1.10.0", "[RPG-DiffusionMaster] install openai")
if not launch.is_installed("google-generativeai"):
    launch.run_pip(
        "install google-generativeai==0.3.2",
        "[RPG-DiffusionMaster] install google-generativeai",
    )
if not launch.is_installed("transformers"):
    launch.run_pip("install transformers", "[RPG-DiffusionMaster] install transformers")
