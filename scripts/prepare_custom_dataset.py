import argparse
import os
import random
import shutil
from pathlib import Path


VIDEO_EXTS = {".mp4", ".mov", ".gif"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def is_supported_item(p: Path) -> bool:
    if p.is_dir():
        return True
    if p.suffix.lower() in VIDEO_EXTS:
        return True
    if p.suffix.lower() in IMAGE_EXTS:
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare data/custom-256/{train,test} from a flat folder of videos or frame-folders."
    )
    parser.add_argument("--input", required=True, help="Folder containing .mp4/.mov/.gif, frame folders, or stacked images")
    parser.add_argument("--output", default="data/custom-256", help="Output dataset root (will create train/test)")
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=["copy", "symlink"],
        default="symlink",
        help="Use symlinks (fast, saves space) or copy (portable).",
    )
    args = parser.parse_args()

    inp = Path(args.input).expanduser().resolve()
    out_root = Path(args.output).expanduser()
    out_train = out_root / "train"
    out_test = out_root / "test"

    if not inp.exists() or not inp.is_dir():
        raise SystemExit(f"Input must be an existing directory: {inp}")

    items = [p for p in sorted(inp.iterdir()) if is_supported_item(p)]
    if not items:
        raise SystemExit(
            "No supported items found. Put videos (.mp4/.mov/.gif), folders of frames, or stacked images in the input folder."
        )

    random.Random(args.seed).shuffle(items)
    n_test = max(1, int(round(len(items) * args.test_ratio))) if len(items) >= 5 else max(1, int(len(items) * args.test_ratio))
    test_items = items[:n_test]
    train_items = items[n_test:]
    if not train_items:
        train_items = test_items[:-1]
        test_items = test_items[-1:]

    out_train.mkdir(parents=True, exist_ok=True)
    out_test.mkdir(parents=True, exist_ok=True)

    def link_or_copy(src: Path, dst: Path) -> None:
        if dst.exists() or dst.is_symlink():
            return
        if args.mode == "symlink":
            os.symlink(src, dst, target_is_directory=src.is_dir())
        else:
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    for p in train_items:
        link_or_copy(p, out_train / p.name)
    for p in test_items:
        link_or_copy(p, out_test / p.name)

    print(f"Prepared dataset at: {out_root}")
    print(f"Train items: {len(train_items)}  Test items: {len(test_items)}")
    print(f"Mode: {args.mode}")


if __name__ == "__main__":
    main()

