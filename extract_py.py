from pathlib import Path

# Thư mục hiện tại
folder = Path.cwd() / "scripts"
folder_name = folder.name

output_file = folder / f"{folder_name}.txt"

py_files = sorted(folder.glob("*.py"))

with output_file.open("w", encoding="utf-8") as out:
    out.write(f"# Folder: {folder_name}\n\n")

    for py_file in py_files:
        out.write("=" * 80 + "\n")
        out.write(f"# File: {py_file.name}\n")
        out.write("=" * 80 + "\n\n")

        try:
            out.write(py_file.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            out.write(py_file.read_text(encoding="utf-8", errors="ignore"))

        out.write("\n\n")

print(f"Done! Exported to: {output_file.name}")
