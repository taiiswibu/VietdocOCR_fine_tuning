"""Prepare gated Hugging Face OCR dataset as PNG + JSONL.

Credentials are read only from HF_TOKEN / Kaggle Secrets.

Research guarantees:
- official test remains held out
- exact RGB duplicates are removed across splits
- train/validation are grouped by normalized transcript
- vocabulary additions are learned from TRAIN ONLY
- rare/unexpected new characters are audited
"""

import argparse
import hashlib
import json
import os
import unicodedata

from collections import Counter
from pathlib import Path

import yaml
from PIL import Image, ImageOps


def normalize_text(text: str) -> str:
    """Normalize OCR transcript without destroying Vietnamese diacritics."""
    text = unicodedata.normalize("NFC", str(text))

    # Remove Unicode control/format characters.
    text = "".join(
        ch
        for ch in text
        if not unicodedata.category(ch).startswith("C")
    )

    # Collapse tabs/newlines/multiple spaces.
    return " ".join(text.split())


def split_for(text: str, seed: int, val_fraction: float):
    """Assign identical normalized transcripts to the same split."""
    group = normalize_text(text)

    digest = hashlib.sha256(
        f"{seed}:{group}".encode("utf-8")
    ).hexdigest()

    value = int(digest[:8], 16) / 2**32

    split = "val" if value < val_fraction else "train"

    return split, digest


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/handwriting"),
    )

    parser.add_argument(
        "--dataset",
        default="5CD-AI/Viet-Handwriting-OCR-v2",
    )

    parser.add_argument(
        "--revision",
        default="main",
    )

    parser.add_argument("--image-column")
    parser.add_argument("--text-column")

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--limit-train",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vietocr.yml"),
    )

    # NEW: vocabulary research guards
    parser.add_argument(
        "--min-new-char-frequency",
        type=int,
        default=5,
        help=(
            "Minimum TRAIN frequency required before a character "
            "may be appended to the VietOCR vocabulary."
        ),
    )

    parser.add_argument(
        "--max-new-chars",
        type=int,
        default=64,
        help=(
            "Maximum number of vocabulary characters that may be "
            "added. Preparation fails if this guard is exceeded."
        ),
    )

    parser.add_argument(
        "--accept-noncommercial",
        action="store_true",
    )

    args = parser.parse_args()

    # ---------- argument validation ----------

    if not args.accept_noncommercial:
        parser.error(
            "Dataset công bố CC BY-NC 4.0. "
            "Đọc điều kiện và thêm --accept-noncommercial "
            "cho dự án phi thương mại."
        )

    if not 0 < args.val_fraction < 0.5:
        parser.error(
            "val-fraction phải nằm trong (0, 0.5)"
        )

    if args.min_new_char_frequency < 1:
        parser.error(
            "--min-new-char-frequency phải >= 1"
        )

    if args.max_new_chars < 0:
        parser.error(
            "--max-new-chars phải >= 0"
        )

    manifests = [
        args.output / f"{split}.jsonl"
        for split in ("train", "val", "test")
    ]

    if any(path.exists() for path in manifests):
        parser.error(
            "Thư mục đã có manifest; dùng thư mục mới "
            "để không ghi đè lần chuẩn bị dữ liệu trước."
        )

    # Lazy imports because these packages are expensive.
    from datasets import load_dataset
    from huggingface_hub import HfApi

    token = os.getenv("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "HF_TOKEN chưa được thiết lập."
        )

    # Resolve mutable `main` to immutable dataset commit.
    revision = HfApi().dataset_info(
        args.dataset,
        revision=args.revision,
        token=token,
    ).sha

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    (args.output / "images").mkdir(
        exist_ok=True
    )

    files = {
        split: (
            args.output / f"{split}.jsonl"
        ).open(
            "w",
            encoding="utf-8",
        )
        for split in ("train", "val", "test")
    }

    counts = Counter()

    rejected = []

    # Exact image hash across ALL splits.
    seen = set()

    # IMPORTANT:
    # Character statistics are TRAIN ONLY.
    train_char_frequency = Counter()

    columns = None

    try:
        # Test first so duplicates from train matching official test
        # are removed from the train/validation pool.
        for source_split in ("test", "train"):

            dataset = load_dataset(
                args.dataset,
                split=source_split,
                revision=revision,
                streaming=True,
                token=token,
            )

            for index, row in enumerate(dataset):

                if (
                    source_split == "train"
                    and args.limit_train
                    and index >= args.limit_train
                ):
                    break

                if columns is None:
                    image_col = (
                        args.image_column
                        or next(
                            (
                                key
                                for key, value
                                in row.items()
                                if isinstance(
                                    value,
                                    Image.Image,
                                )
                            ),
                            None,
                        )
                    )

                    text_col = (
                        args.text_column
                        or next(
                            (
                                key
                                for key in (
                                    "text",
                                    "label",
                                    "transcription",
                                    "sentence",
                                    "ground_truth",
                                )
                                if isinstance(
                                    row.get(key),
                                    str,
                                )
                            ),
                            None,
                        )
                    )

                    if not image_col or not text_col:
                        raise ValueError(
                            "Không xác định được cột. "
                            f"Các cột: {list(row)}. "
                            "Đặt --image-column / --text-column."
                        )

                    columns = (
                        image_col,
                        text_col,
                    )

                try:
                    image = ImageOps.exif_transpose(
                        row[columns[0]]
                    ).convert("RGB")

                    text = normalize_text(
                        row[columns[1]]
                    )

                    if (
                        not text
                        or image.width < 3
                        or image.height < 3
                    ):
                        raise ValueError(
                            "empty_text_or_tiny_image"
                        )

                    # Exact decoded RGB image fingerprint.
                    digest = hashlib.sha256(
                        str(image.size).encode("utf-8")
                        + image.tobytes()
                    ).hexdigest()

                    if digest in seen:
                        counts[
                            "duplicates_removed"
                        ] += 1
                        continue

                    seen.add(digest)

                    if source_split == "test":
                        destination = "test"
                        group = digest
                    else:
                        destination, group = split_for(
                            text,
                            args.seed,
                            args.val_fraction,
                        )

                    name = (
                        f"images/"
                        f"{source_split}-"
                        f"{index:07d}.png"
                    )

                    image.save(
                        args.output / name
                    )

                    record = {
                        "id": (
                            f"{source_split}-{index}"
                        ),
                        "image": name,
                        "text": text,
                        "sha256": digest,
                        "group": group,
                        "source_split": source_split,
                    }

                    files[destination].write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    counts[destination] += 1

                    # Vocabulary statistics MUST NOT use val/test.
                    if destination == "train":
                        train_char_frequency.update(
                            text
                        )

                    if index % 2000 == 0:
                        print(
                            source_split,
                            index,
                            dict(counts),
                            flush=True,
                        )

                except Exception as exc:
                    rejected.append({
                        "split": source_split,
                        "index": index,
                        "reason": str(exc),
                    })

    finally:
        for file_handle in files.values():
            file_handle.close()

    # ---------- Vocabulary audit ----------

    base = yaml.safe_load(
        args.config.read_text(
            encoding="utf-8"
        )
    )

    raw_vocab = base["vocab"]

    if isinstance(raw_vocab, list):
        base_vocab = "".join(raw_vocab)
    else:
        base_vocab = str(raw_vocab)

    base_vocab_chars = set(base_vocab)

    missing_chars = sorted(
        char
        for char in train_char_frequency
        if char not in base_vocab_chars
    )

    appended_vocab_audit = []

    eligible_chars = []

    for char in missing_chars:
        frequency = int(
            train_char_frequency[char]
        )

        included = (
            frequency
            >= args.min_new_char_frequency
        )

        if included:
            eligible_chars.append(char)

        appended_vocab_audit.append({
            "character": char,
            "codepoint": (
                f"U+{ord(char):04X}"
            ),
            "unicode_name": (
                unicodedata.name(
                    char,
                    "UNKNOWN",
                )
            ),
            "train_frequency": frequency,
            "included": included,
            "reason": (
                "frequency_threshold_met"
                if included
                else "below_min_frequency"
            ),
        })

    additions = "".join(
        eligible_chars
    )

    # Report is written BEFORE the guard raises so the notebook
    # can show exactly which characters caused the problem.
    report = {
        "dataset": args.dataset,
        "revision": revision,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "columns": columns,
        "counts": dict(counts),
        "rejected": rejected,
        "appended_vocab": additions,
        "appended_vocab_audit": (
            appended_vocab_audit
        ),
        "min_new_char_frequency": (
            args.min_new_char_frequency
        ),
        "max_new_chars": (
            args.max_new_chars
        ),
        "limited_train": (
            args.limit_train
        ),
        "split_policy": (
            "Official test reserved first; "
            "exact RGB duplicates removed across splits; "
            "train/val grouped by normalized transcription. "
            "Vocabulary learned from train only. "
            "Writer-disjointness unverified."
        ),
    }

    report_path = (
        args.output / "preparation.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if len(eligible_chars) > args.max_new_chars:
        raise RuntimeError(
            "Vocabulary expansion guard failed: "
            f"{len(eligible_chars)} characters passed "
            "the minimum-frequency threshold, "
            f"but --max-new-chars={args.max_new_chars}. "
            "Inspect preparation.json -> "
            "appended_vocab_audit before changing the guard."
        )

    # ---------- Final VietOCR config ----------

    base["vocab"] = (
        base_vocab + additions
    )

    (
        args.output / "config.yml"
    ).write_text(
        yaml.safe_dump(
            base,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    # ---------- final guards ----------

    if not all(
        counts[split]
        for split in (
            "train",
            "val",
            "test",
        )
    ):
        raise RuntimeError(
            "Một hoặc nhiều split rỗng: "
            f"{dict(counts)}"
        )

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "revision": revision,
                "counts": dict(counts),
                "duplicates_removed": (
                    counts[
                        "duplicates_removed"
                    ]
                ),
                "rejected": len(rejected),
                "new_vocab_chars": (
                    len(additions)
                ),
                "output": str(
                    args.output
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()