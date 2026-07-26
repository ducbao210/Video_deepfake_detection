from pathlib import Path

root = Path("data/raw")

with open("data/name.txt", "w", encoding="utf-8") as f:
    for folder in root.iterdir():
        if folder.is_dir():
            f.write(f"{folder.name}:\n")
            for file in folder.glob("*.mp4"):
                f.write(f"  {file.name}\n")