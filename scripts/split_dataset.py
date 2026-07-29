import sys
import random
from collections import defaultdict
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import get_logger


def parse_actors(video_id: str) -> frozenset:
    """
    Extract actor IDs from DFD video filenames.

      real:  "06__walk_down_hall_angry"              -> {"06"}
      fake:  "01_02__exit_phone_room__YVGY8LOK"      -> {"01", "02"}
    """

    head = video_id.split("__")[0]
    return frozenset(head.split("_"))


def build_records(video_paths, label, processed_dir):
    records = []
    for video in video_paths:
        num_frames = len(list(video.glob("*.jpg")))
        if num_frames == 0:
            continue
        records.append(
            {
                # .as_posix() ensures forward slashes ("/") are used, even on Windows.
                "video_path": video.relative_to(processed_dir).as_posix(),
                "video_id": video.name,
                "label": label,
                "num_frames": num_frames,
                "actors": "|".join(sorted(parse_actors(video.name))),
            }
        )
    return records


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logger = get_logger(cfg.logging, log_file="split_dataset.log")

    processed_dir = Path(cfg.paths.data.processed_dir)
    split_dir = Path(cfg.paths.data.split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    classes = {
        "original": ("DFD_original_sequences", 0),
        "manipulated": ("DFD_manipulated_sequences", 1),
    }

    # Collect all video records
    all_records = []
    for _, (folder_name, label) in classes.items():
        class_dir = processed_dir / folder_name
        if not class_dir.exists():
            logger.warning(f"Directory not found: {class_dir}")
            continue
        video_paths = sorted(
            [p for p in class_dir.iterdir() if p.is_dir()], key=lambda x: x.name
        )
        all_records.extend(build_records(video_paths, label, processed_dir))

    logger.info(f"Found {len(all_records)} videos with extracted frames.")

    strategy = cfg.split.get("strategy", "actor_disjoint")

    if strategy == "random":
        logger.warning(
            "strategy='random': train/val/test share actors and scenes. "
            "This can lead to overly optimistic metrics. Use only as a baseline."
        )

        rnd = random.Random(cfg.split.random_seed)
        rnd.shuffle(all_records)
        n = len(all_records)
        i1 = int(n * cfg.split.train)
        i2 = i1 + int(n * cfg.split.val)
        buckets = {
            "train": all_records[:i1],
            "val": all_records[i1:i2],
            "test": all_records[i2:],
            "dropped": [],
        }

    else:
        all_actors = sorted({a for r in all_records for a in r["actors"].split("|")})
        logger.info(f"Found {len(all_actors)} unique actors: {all_actors}")
        n_test = cfg.split.get("num_test_actors", 6)
        n_val = cfg.split.get("num_val_actors", 6)

        rnd = random.Random(cfg.split.random_seed)
        shuffled = all_actors[:]
        rnd.shuffle(shuffled)

        test_actors = set(shuffled[:n_test])
        val_actors = set(shuffled[n_test : n_test + n_val])
        train_actors = set(shuffled[n_test + n_val :])

        logger.info(f"Train actors ({len(train_actors)}): {sorted(train_actors)}")
        logger.info(f"Validation actors ({len(val_actors)}): {sorted(val_actors)}")
        logger.info(f"Test actors ({len(test_actors)}): {sorted(test_actors)}")
        buckets = defaultdict(list)
        for r in all_records:
            actors = set(r["actors"].split("|"))
            if actors <= train_actors:
                buckets["train"].append(r)
            elif actors <= val_actors:
                buckets["val"].append(r)
            elif actors <= test_actors:
                buckets["test"].append(r)
            else:
                # Videos containing actors from multiple splits are discarded to prevent data leakage.
                buckets["dropped"].append(r)

    # Save split files
    for name in ("train", "val", "test"):
        df = pd.DataFrame(buckets[name]).drop(columns=["actors"], errors="ignore")
        df.to_csv(split_dir / f"{name}.csv", index=False)

    if buckets["dropped"]:
        pd.DataFrame(buckets["dropped"]).to_csv(split_dir / "dropped.csv", index=False)

    # Summary
    logger.info("========== SUMMARY ==========")
    logger.info(f"Strategy: {strategy}")
    for name in ("train", "val", "test", "dropped"):
        rows = buckets[name]
        real = sum(1 for r in rows if r["label"] == 0)
        fake = sum(1 for r in rows if r["label"] == 1)
        logger.info(f"{name:8s}: {len(rows):5d} video (real={real:4d}, fake={fake:4d})")

    kept = sum(len(buckets[n]) for n in ("train", "val", "test"))
    logger.info(
        f"Retained {kept}/{len(all_records)} videos ({kept/len(all_records):.1%})"
    )
    logger.info("=============================")

    # Check for actor leakage (automatically logged for reporting)
    def actor_set(name):
        return {a for r in buckets[name] for a in r["actors"].split("|")}

    overlap_test = actor_set("train") & actor_set("test")
    overlap_val = actor_set("train") & actor_set("val")
    logger.info(f"[LEAK CHECK] Train ∩ Test actors: {sorted(overlap_test) or 'NONE'}")

    logger.info(
        f"[LEAK CHECK] Train ∩ Validation actors: {sorted(overlap_val) or 'NONE'}"
    )


if __name__ == "__main__":
    main()
