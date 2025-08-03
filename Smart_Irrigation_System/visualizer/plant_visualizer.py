import matplotlib.pyplot as plt
from PIL import Image
import os

def get_health_state(yield_potential: float) -> str:
    """
    Categorize plant health based on yield potential.
    """
    if yield_potential >= 1.5:
        return "healthy"
    elif yield_potential >= 0.8:
        return "stressed"
    else:
        return "dying"

def load_crop_image(yield_potential: float):
    """
    Loads a plant image corresponding to the current health state without displaying it.

    Returns:
    - PIL.Image object or None if the image is not found.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(BASE_DIR, "plants", f"{get_health_state(yield_potential)}.png")

    try:
        img = Image.open(img_path)
        return img
    except FileNotFoundError:
        print(f"⚠️ Missing image for state: {get_health_state(yield_potential)} at {img_path}")
        return None

def show_crop_image(yield_potential: float, day: int):
    """
    Displays a plant image corresponding to the current health state.
    """
    img = load_crop_image(yield_potential)
    if img is not None:
        plt.figure(figsize=(3, 3))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Day {day}: {get_health_state(yield_potential).capitalize()} Crop (Yield Potential = {yield_potential:.2f})")
        plt.show()

def load_growth_stage_image(stage_name: str):
    """
    Loads a plant image corresponding to the growth stage.
    
    Parameters:
    - stage_name: Name of the growth stage (Seedling, Jointing, Staminate, Filling, Maturity)
    
    Returns:
    - PIL.Image object or None if the image is not found.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(BASE_DIR, "plants", f"{stage_name}.jpg")
    
    try:
        img = Image.open(img_path)
        return img
    except FileNotFoundError:
        print(f"⚠️ Missing image for growth stage: {stage_name} at {img_path}")
        return None