import os, shutil, random
from pathlib import Path


def make_validation_set(
    train_dir: Path, val_dir: Path, val_fraction: float, seed: int
) -> None:
    if any(Path(val_dir / "pos").iterdir()):
        reset_validation_set(train_dir=train_dir, val_dir=val_dir)

    for category in ("neg", "pos"):
        files = os.listdir(train_dir / category)
        random.Random(seed).shuffle(files)
        num_val_samples = int(val_fraction * len(files))
        val_files = files[-num_val_samples:]
        for fname in val_files:
            shutil.move(train_dir / category / fname, val_dir / category / fname)


def reset_validation_set(train_dir: Path, val_dir: Path) -> None:
    for category in ("neg", "pos"):
        files = os.listdir(val_dir / category)
        files.remove('.gitignore')
        for fname in files:
            shutil.move(val_dir / category / fname, train_dir / category / fname)
