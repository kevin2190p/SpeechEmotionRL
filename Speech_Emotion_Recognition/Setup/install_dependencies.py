# install_dependencies.py
import subprocess
import sys

required_packages = [
    "librosa",
    "stable-baselines3[extra]",
    "torch",
    "torchaudio",
    "scikit-learn",
    "matplotlib",
    "numpy",
    "tqdm",
    "gymnasium",
    "sb3-contrib",
    "transformers",
    "shap",
    "shimmy",
    "optuna",
    "SpeechRecognition",
    "seaborn",
    "pydub",
    "flask",
    "Pillow",
    "ffmpeg-python",
    "werkzeug",
    "psutil"
]

def install_packages(packages):
    for pkg in packages:
        print(f"\nInstalling: {pkg}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {pkg}: {e}")

if __name__ == "__main__":
    install_packages(required_packages)
    print("\nAll dependencies attempted for installation.")
