"""
Containt function download data
"""
import os 
import zipfile
import requests
from pathlib import Path

# setup path to data folder
data_path = Path("data")
image_path = data_path / "pizza_steak_sushi"

# If the image folder doesn'exist, download it and prepare it
if image_path.is_dir():
    print(f"{image_path} directory exists.")
else:
    print(f"Did not find {image_path} directory, creating...")
    image_path.mkdir(parents=True, exist_ok=True)

# Dowload pizza, steak, sushi data
with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    print("Downloading pizza, steak, sushi data...")
    f.write(request.content)

with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
    print("Unziping pizza, steak, sushi data...")
    zip_ref.extractall(image_path)

# remove zip file
os.remove(data_path / "pizza_steak_sushi.zip")
