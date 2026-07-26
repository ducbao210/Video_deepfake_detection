import re
from pathlib import Path


def extract_epoch_from_filename(checkpoint_path: str | Path) -> int:
    checkpoint_path = Path(checkpoint_path)

    # Validate extension
    if checkpoint_path.suffix not in {".pt", ".pth"}:
        raise ValueError(
            f"Checkpoint must have '.pt' or '.pth' extension, got '{checkpoint_path.suffix}'."
        )

    filename = checkpoint_path.stem

    match = re.search(r"epoch[_-]?(\d+)", filename, re.IGNORECASE)
    if not match:
        raise ValueError(
            f"Cannot extract epoch from checkpoint filename '{checkpoint_path.name}'."
        )

    return int(match.group(1))


if __name__ == "__main__":
    print(extract_epoch_from_filename("output/model/epoch_10_2026.pth"))
