import launch

if not launch.is_installed("openai"):
    launch.run_pip("install openai==1.10.0", "requirements RPG-DiffusionMaster")
if not launch.is_installed("google-generativeai"):
    launch.run_pip("install google-generativeai==0.3.2", "requirements RPG-DiffusionMaster")
