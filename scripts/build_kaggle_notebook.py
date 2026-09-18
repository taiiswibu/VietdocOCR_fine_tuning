"""Generate the clean, output-free, research-grade Kaggle fine-tuning notebook."""
import json
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


cells = [
markdown(r"""# VietOCR × 5CD — Fine-tuning có thể tái lập trên Kaggle T4

Notebook này thực hiện **một thí nghiệm mới hoàn toàn**:

1. lấy checkpoint VietOCR pretrained gốc làm trạng thái **trước fine-tune**;
2. fine-tune chính checkpoint đó trên train split của `5CD-AI/Viet-Handwriting-OCR-v2`;
3. chọn epoch bằng validation CER;
4. chỉ sau khi chọn model mới đánh giá hai checkpoint trên **cùng official test manifest**;
5. báo cáo CER, whitespace-WER, Exact Match, khoảng tin cậy bootstrap và phân tích lỗi.

> **Không dùng** `vudang449/vietocr-vietnamese-handwriting` hay bất kỳ model community nào làm đối chứng. Đây là phép so sánh paired: **VietOCR pretrained trước update** và **VietOCR do chính run này fine-tune trên 5CD**.

### Câu hỏi nghiên cứu

> Fine-tuning VietOCR pretrained trên train split của 5CD có cải thiện nhận dạng dòng chữ viết tay tiếng Việt trên official test split hay không?

CER/WER càng thấp càng tốt; Exact Match càng cao càng tốt. Notebook không giả định fine-tune chắc chắn sẽ tốt hơn và không điền sẵn kết quả.
"""),
markdown(r"""## 0. Thiết kế thí nghiệm

| Thành phần | Quy ước |
|---|---|
| Model trước fine-tune | VietOCR `vgg_transformer` pretrained, có revision và SHA-256 |
| Model sau fine-tune | `best-finetuned.pth`: epoch đã nhận optimizer update và có validation CER thấp nhất |
| Model triển khai | `model.pth`: tốt nhất giữa pretrained initialization và các epoch fine-tune |
| Train/validation | Tách từ official train; cùng transcript chuẩn hóa không xuất hiện ở cả hai tập |
| Test | Official test, giữ khóa đến khi đã chọn checkpoint |
| Model selection | Validation CER; tuyệt đối không chọn epoch theo test |
| So sánh cuối | Cùng test rows, cùng normalization, cùng device và decoder |
| Seed | Ghi cố định trong config và provenance |

Việc giữ riêng `best-finetuned.pth` và `model.pth` rất quan trọng: nếu fine-tuning làm giảm chất lượng, ta vẫn đo được checkpoint đã fine-tune tốt nhất, nhưng không nhầm nó với model nên triển khai.
"""),
code(r"""# ===== Experiment configuration: edit only this cell before a new run =====
CFG = {
    "dataset": "5CD-AI/Viet-Handwriting-OCR-v2",
    "dataset_revision": "main",  # resolved to an immutable commit by prepare.py
    "seed": 42,
    "val_fraction": 0.05,
    "min_new_char_frequency": 5,
    "max_new_chars": 64,
    "epochs": 8,
    "batch_size": 8,
    "gradient_accumulation": 4,   # effective batch size = 32
    "learning_rate": 3e-5,
    "patience": 3,
    "validation_samples": 0,      # 0 = full validation set
    "save_every_updates": 100,
    "limit_train": 0,             # 0 = full train split; never use a limit for final results
    "run_name": "vietocr_5cd_seed42_lr3e-5_v1",
    "bootstrap_repeats": 2000,
}

RUN_TRAINING = True
RESUME_INTERRUPTED_RUN = False  # keep False for a genuinely fresh run
PROJECT_INPUT = ""               # optional: /kaggle/input/<your-project-dataset>
REPO_URL = "https://github.com/taiiswibu/VietdocOCR_fine_tuning.git"

assert CFG["limit_train"] == 0, "Final experiment must use the complete training split."
assert 0 < CFG["val_fraction"] < 0.5
assert CFG["epochs"] >= 1 and CFG["batch_size"] >= 1
assert CFG["learning_rate"] > 0
print(CFG)
"""),
markdown(r"""## 1. Khởi tạo project và môi trường

Khuyến nghị trên Kaggle:

- Accelerator: **GPU T4**;
- Internet: bật khi cài package/tải checkpoint;
- Secret: `HF_TOKEN` read-only đã được cấp quyền truy cập dataset gated 5CD;
- Add Input: source project hiện tại để revision code không đổi giữa các lần chạy.

Nếu không có source project trong Kaggle Input, cell dưới sẽ clone repository và ghi lại Git commit. Để tái lập chặt hơn, nên dùng Kaggle Dataset chứa source đã chốt.
"""),
code(r"""from pathlib import Path
import os, sys, json, shutil, subprocess, hashlib, random, platform, time

KAGGLE = Path("/kaggle/input").exists()
WORK_PARENT = Path("/kaggle/working") if KAGGLE else Path.cwd()
ROOT = WORK_PARENT / "vietdoc-ocr"

def is_project(path: Path) -> bool:
    return (path / "pyproject.toml").exists() and (path / "training/train.py").exists()

if not is_project(ROOT):
    source = Path(PROJECT_INPUT) if PROJECT_INPUT else None
    if source and not is_project(source):
        raise FileNotFoundError(f"PROJECT_INPUT is not a VietDoc source tree: {source}")
    if source is None and KAGGLE:
        candidates = sorted({p.parent for p in Path("/kaggle/input").rglob("pyproject.toml") if is_project(p.parent)})
        if len(candidates) == 1:
            source = candidates[0]
        elif len(candidates) > 1:
            raise RuntimeError(f"Multiple project inputs found; set PROJECT_INPUT explicitly: {candidates}")
    if source:
        shutil.copytree(source, ROOT)
        print("Copied source:", source)
    else:
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(ROOT)], check=True)
        print("Cloned source:", REPO_URL)

os.chdir(ROOT)
train_source = (ROOT / "training/train.py").read_text(encoding="utf-8")
assert "best-finetuned.pth" in train_source, (
    "Source project is older than this notebook. Update training/train.py so it preserves "
    "the best fine-tuned epoch separately from the deployment-selected checkpoint."
)

def shell(*args):
    return subprocess.run([str(x) for x in args], check=True)

try:
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    git_commit = "unavailable"
print("ROOT:", ROOT)
print("CODE COMMIT:", git_commit)
"""),
code(r"""# Install only what the research pipeline needs.
shell(sys.executable, "-m", "pip", "install", "-q", "-e", ".[train,test]",
      "vietocr==0.3.13", "einops==0.2.0", "gdown==4.4.0",
      "pandas>=2.0,<3", "seaborn>=0.13,<1")

# Fail early if project invariants are broken.
shell(sys.executable, "-m", "pytest", "-q")
"""),
code(r"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from IPython.display import display, FileLink
import torch

random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG["seed"])
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

assert torch.cuda.is_available(), "Kaggle Settings → Accelerator → GPU T4, rồi restart session."
DEVICE = "cuda:0"
gpu_name = torch.cuda.get_device_name(0)
print({
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": gpu_name,
    "platform": platform.platform(),
})
"""),
code(r"""# Load the gated-dataset credential without printing it.
if KAGGLE:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
assert os.environ.get("HF_TOKEN"), "Create a read-only Kaggle Secret named HF_TOKEN."
print("HF_TOKEN loaded:", bool(os.environ.get("HF_TOKEN")))
"""),
code(r"""DATA_DIR = ROOT / "data" / f"5cd-seed{CFG['seed']}-val{int(CFG['val_fraction']*100)}"
RUN_DIR = ROOT / "runs" / CFG["run_name"]
BASE_DIR = ROOT / "models" / "base"
REPORT_DIR = ROOT / "runs" / f"{CFG['run_name']}-report"
PLOT_DIR = REPORT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

print("DATA_DIR  =", DATA_DIR)
print("RUN_DIR   =", RUN_DIR)
print("REPORT_DIR=", REPORT_DIR)
"""),
markdown(r"""## 2. Khóa checkpoint VietOCR gốc

Cell này chỉ tải `--base`. Không tải model handwriting community. `provenance.json` và SHA-256 giúp chứng minh checkpoint “trước fine-tune” thực sự là model gốc đã khai báo.
"""),
code(r"""shell(sys.executable, "scripts/download_models.py", "--base")
base_provenance = json.loads((BASE_DIR / "provenance.json").read_text(encoding="utf-8"))
base_sha = hashlib.sha256((BASE_DIR / "model.pth").read_bytes()).hexdigest()

assert base_provenance["kind"] == "external_pretrained"
assert base_sha == base_provenance["sha256"]
assert "vudang449" not in json.dumps(base_provenance).lower()
display(pd.DataFrame([{
    "checkpoint": "VietOCR pretrained (before fine-tune)",
    "repository": base_provenance["repository"],
    "revision": base_provenance["revision"],
    "sha256": base_sha,
}]))
"""),
markdown(r"""## 3. Chuẩn bị 5CD mà không làm rò rỉ test

Quy trình chuẩn bị:

- resolve `dataset_revision` thành commit bất biến;
- NFC-normalize transcript, bỏ control characters và chuẩn hóa whitespace;
- dành official test làm held-out test;
- loại ảnh RGB trùng lặp xuyên split bằng SHA-256;
- tách train/validation theo hash của transcript chuẩn hóa;
- chỉ học vocabulary bổ sung từ **train**, không nhìn test;
- dừng nếu vocabulary tăng bất thường.

Dataset là gated. Bạn phải đọc và chấp nhận điều kiện trên trang dataset trước khi chạy.
"""),
code(r"""required_manifests = [DATA_DIR / f"{name}.jsonl" for name in ("train", "val", "test")]
if not all(path.exists() for path in required_manifests):
    cmd = [
        sys.executable, "-m", "training.prepare",
        "--dataset", CFG["dataset"],
        "--revision", CFG["dataset_revision"],
        "--output", DATA_DIR,
        "--seed", CFG["seed"],
        "--val-fraction", CFG["val_fraction"],
        "--min-new-char-frequency", CFG["min_new_char_frequency"],
        "--max-new-chars", CFG["max_new_chars"],
        "--limit-train", CFG["limit_train"],
        "--accept-noncommercial",
    ]
    completed = subprocess.run([str(x) for x in cmd])
    if completed.returncode != 0:
        report_path = DATA_DIR / "preparation.json"
        if report_path.exists():
            display(json.loads(report_path.read_text(encoding="utf-8")))
        raise RuntimeError("Dataset preparation failed. Read preparation.json before changing any guard.")
else:
    print("Prepared manifests already exist; validating them instead of overwriting.")
"""),
code(r"""prep = json.loads((DATA_DIR / "preparation.json").read_text(encoding="utf-8"))
assert prep["dataset"] == CFG["dataset"]
assert prep["seed"] == CFG["seed"]
assert abs(prep["val_fraction"] - CFG["val_fraction"]) < 1e-12
assert prep["limited_train"] == 0
assert len(prep["appended_vocab"]) <= CFG["max_new_chars"], (
    "Vocabulary expansion exceeded the research guard. Audit appended_vocab_audit; "
    "do not simply raise the limit."
)

summary = pd.DataFrame([{
    "dataset": prep["dataset"],
    "resolved revision": prep["revision"],
    "train": prep["counts"].get("train", 0),
    "validation": prep["counts"].get("val", 0),
    "official test": prep["counts"].get("test", 0),
    "duplicates removed": prep["counts"].get("duplicates_removed", 0),
    "rejected": len(prep["rejected"]),
    "new vocabulary chars": len(prep["appended_vocab"]),
}])
display(summary)

vocab_audit = pd.DataFrame(prep["appended_vocab_audit"])
if len(vocab_audit):
    display(vocab_audit.sort_values("train_frequency", ascending=False).reset_index(drop=True))
else:
    print("No vocabulary expansion was required.")
"""),
markdown(r"""### 3.1 Kiểm tra split và leakage

Official test không được dùng để chọn hyperparameter. Train/validation không được có cùng ảnh hoặc cùng transcript-group. Nếu một assertion thất bại, dừng thí nghiệm và sửa dữ liệu thay vì bỏ assertion.
"""),
code(r"""def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

rows = {name: read_jsonl(DATA_DIR / f"{name}.jsonl") for name in ("train", "val", "test")}
for name, split_rows in rows.items():
    assert split_rows, f"Empty split: {name}"

sha = {name: {r["sha256"] for r in split_rows} for name, split_rows in rows.items()}
assert not (sha["train"] & sha["val"])
assert not (sha["train"] & sha["test"])
assert not (sha["val"] & sha["test"])

train_groups = {r["group"] for r in rows["train"]}
val_groups = {r["group"] for r in rows["val"]}
assert not (train_groups & val_groups), "Transcript-group leakage between train and validation."
assert all(r["source_split"] == "test" for r in rows["test"])

integrity = pd.DataFrame([
    {"check": "Image SHA overlap train↔val", "value": len(sha["train"] & sha["val"]), "expected": 0},
    {"check": "Image SHA overlap train↔test", "value": len(sha["train"] & sha["test"]), "expected": 0},
    {"check": "Image SHA overlap val↔test", "value": len(sha["val"] & sha["test"]), "expected": 0},
    {"check": "Transcript-group overlap train↔val", "value": len(train_groups & val_groups), "expected": 0},
])
display(integrity)
"""),
markdown(r"""## 4. EDA chỉ trên train/validation

Không xem mẫu test ở giai đoạn này. EDA tập trung vào độ dài nhãn và hình học ảnh vì chúng ảnh hưởng trực tiếp đến sequence length, padding và chi phí Transformer.
"""),
code(r"""eda_records = []
for split in ("train", "val"):
    for row in rows[split]:
        eda_records.append({
            "split": split,
            "characters": len(row["text"]),
            "words": len(row["text"].split()),
        })
eda = pd.DataFrame(eda_records)
display(eda.groupby("split")[["characters", "words"]].describe(percentiles=[.5, .9, .95, .99]).round(2))

sns.set_theme(style="whitegrid", context="notebook")
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
sns.histplot(data=eda, x="characters", hue="split", bins=50, element="step", stat="density", common_norm=False, ax=axes[0])
axes[0].set_title("Độ dài transcript"); axes[0].set_xlabel("Số ký tự")
sns.histplot(data=eda, x="words", hue="split", bins=35, element="step", stat="density", common_norm=False, ax=axes[1])
axes[1].set_title("Số từ mỗi dòng"); axes[1].set_xlabel("Số từ")
fig.suptitle("5CD train/validation — label distribution", fontweight="bold")
fig.tight_layout()
fig.savefig(PLOT_DIR / "eda-label-lengths.png", dpi=180, bbox_inches="tight")
plt.show()
"""),
code(r"""# Inspect a deterministic sample of train/validation image geometry.
rng = np.random.default_rng(CFG["seed"])
geometry_records = []
for split in ("train", "val"):
    sample_n = min(1200, len(rows[split]))
    for idx in rng.choice(len(rows[split]), size=sample_n, replace=False):
        row = rows[split][int(idx)]
        with Image.open(DATA_DIR / row["image"]) as image:
            width, height = image.size
        geometry_records.append({"split": split, "width": width, "height": height, "aspect_ratio": width / height})
geometry = pd.DataFrame(geometry_records)
display(geometry.groupby("split")[["width", "height", "aspect_ratio"]].describe(percentiles=[.5, .9, .99]).round(2))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
sns.scatterplot(data=geometry, x="width", y="height", hue="split", alpha=.4, s=22, ax=axes[0])
axes[0].set_title("Kích thước ảnh mẫu")
sns.histplot(data=geometry, x="aspect_ratio", hue="split", bins=50, element="step", stat="density", common_norm=False, ax=axes[1])
axes[1].set_title("Tỷ lệ width / height")
fig.tight_layout()
fig.savefig(PLOT_DIR / "eda-image-geometry.png", dpi=180, bbox_inches="tight")
plt.show()
"""),
code(r"""# Qualitative train samples — never test samples.
sample_indices = rng.choice(len(rows["train"]), size=min(8, len(rows["train"])), replace=False)
fig, axes = plt.subplots(4, 2, figsize=(15, 10))
for ax, idx in zip(axes.flat, sample_indices):
    row = rows["train"][int(idx)]
    with Image.open(DATA_DIR / row["image"]) as image:
        ax.imshow(image.convert("RGB"))
    title = row["text"] if len(row["text"]) <= 90 else row["text"][:87] + "…"
    ax.set_title(title, fontsize=9); ax.axis("off")
fig.suptitle("Train samples after normalization", fontweight="bold")
fig.tight_layout()
fig.savefig(PLOT_DIR / "eda-train-samples.png", dpi=180, bbox_inches="tight")
plt.show()
"""),
markdown(r"""## 5. Fine-tune từ VietOCR pretrained — fresh run

Thiết lập mặc định cho T4:

- full-model AdamW, learning rate `3e-5`;
- mixed precision;
- batch 8 × accumulation 4 = effective batch 32;
- 5% linear warm-up + cosine decay;
- label smoothing 0.1 và gradient clipping 1.0;
- full validation mỗi epoch;
- early stopping theo validation CER;
- checkpoint có optimizer, scheduler, scaler và RNG để resume đúng run bị ngắt.

`RESUME_INTERRUPTED_RUN=False` đảm bảo lần đầu không dùng checkpoint cũ. Nếu Kaggle bị ngắt giữa chừng, save output rồi bật `RESUME_INTERRUPTED_RUN=True`; đó là tiếp tục cùng thí nghiệm, không phải một run mới.
"""),
code(r"""if RUN_TRAINING:
    if RESUME_INTERRUPTED_RUN:
        resume_path = RUN_DIR / "last.pt"
        assert resume_path.exists(), f"Missing resume checkpoint: {resume_path}"
    else:
        if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
            raise FileExistsError(
                f"Fresh run refuses to reuse non-empty {RUN_DIR}. "
                "Change CFG['run_name']; do not delete evidence from a previous experiment."
            )

    train_cmd = [
        sys.executable, "-m", "training.train",
        "--data", DATA_DIR,
        "--output", RUN_DIR,
        "--initial", BASE_DIR / "model.pth",
        "--epochs", CFG["epochs"],
        "--batch-size", CFG["batch_size"],
        "--accum", CFG["gradient_accumulation"],
        "--lr", CFG["learning_rate"],
        "--seed", CFG["seed"],
        "--device", DEVICE,
        "--val-samples", CFG["validation_samples"],
        "--save-every", CFG["save_every_updates"],
        "--patience", CFG["patience"],
    ]
    if RESUME_INTERRUPTED_RUN:
        train_cmd += ["--resume", RUN_DIR / "last.pt"]

    print("Launching:", " ".join(map(str, train_cmd)))
    subprocess.run([str(x) for x in train_cmd], check=True)
else:
    print("Training disabled. Existing RUN_DIR must contain a completed compatible run.")
"""),
markdown(r"""## 6. Kiểm tra model selection và training curves

Điểm epoch 0 là pretrained initialization được đánh giá trên validation trước optimizer step. Các epoch 1…N là checkpoint sau fine-tuning. Đường CER/WER không phải accuracy; thấp hơn mới tốt hơn.
"""),
code(r"""run_meta = json.loads((RUN_DIR / "run.json").read_text(encoding="utf-8"))
initial_val = json.loads((RUN_DIR / "initial-validation.json").read_text(encoding="utf-8"))
history = pd.DataFrame(read_jsonl(RUN_DIR / "metrics.jsonl"))

assert run_meta["training_complete"], "Training is incomplete; resume last.pt instead of reporting results."
assert (RUN_DIR / "best-finetuned.pth").exists(), "No fine-tuned epoch checkpoint was preserved."
assert run_meta["best_finetuned_source"].startswith("epoch-")

selection = pd.DataFrame([
    {"candidate": "Pretrained initialization", "source": "epoch-0", "validation CER": initial_val["cer"]},
    {"candidate": "Best fine-tuned epoch", "source": run_meta["best_finetuned_source"], "validation CER": run_meta["best_finetuned_validation_cer"]},
    {"candidate": "Deployment-selected", "source": run_meta["best_source"], "validation CER": run_meta["best_validation_cer"]},
])
display(selection.style.format({"validation CER": "{:.4%}"}))
display(history)
"""),
code(r"""epochs = history["epoch"].to_numpy()
best_ft_epoch = int(run_meta["best_finetuned_source"].split("-")[-1])

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flat

axes[0].plot(epochs, history["train_loss"], marker="o", color="#2563eb")
axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Cross-entropy")

for ax, key, title, color in [
    (axes[1], "cer", "Validation CER ↓", "#dc2626"),
    (axes[2], "wer_whitespace", "Validation whitespace-WER ↓", "#f97316"),
    (axes[3], "exact_match", "Validation Exact Match ↑", "#059669"),
]:
    ax.plot(np.r_[0, epochs], np.r_[initial_val[key], history[key]], marker="o", color=color)
    ax.axvline(best_ft_epoch, color="#111827", linestyle="--", alpha=.5, label="best fine-tuned")
    ax.set(title=title, xlabel="Epoch", ylabel="Rate")
    ax.legend()

if "learning_rate" in history:
    axes[4].plot(history["updates"], history["learning_rate"], marker="o", color="#7c3aed")
    axes[4].set(title="Learning rate at epoch end", xlabel="Optimizer updates", ylabel="LR")
else:
    axes[4].text(.5, .5, "LR was not logged", ha="center", va="center")
    axes[4].set_axis_off()

axes[5].plot(epochs, history["seconds"] / 60, marker="o", color="#475569")
axes[5].set(title="Validation duration", xlabel="Epoch", ylabel="Minutes")

fig.suptitle("VietOCR fine-tuning diagnostics — train/validation only", fontsize=15, fontweight="bold")
fig.tight_layout()
fig.savefig(PLOT_DIR / "training-curves.png", dpi=180, bbox_inches="tight")
plt.show()
"""),
markdown(r"""## 7. Khóa model sau fine-tune

Checkpoint đối chứng “sau” luôn là epoch fine-tuned tốt nhất theo validation CER, kể cả khi nó kém pretrained. `model.pth` của run vẫn được giữ riêng cho quyết định triển khai.
"""),
code(r"""FINETUNED_DIR = RUN_DIR / "research-finetuned"
FINETUNED_DIR.mkdir(exist_ok=True)
shutil.copy2(RUN_DIR / "best-finetuned.pth", FINETUNED_DIR / "model.pth")
shutil.copy2(RUN_DIR / "config.yml", FINETUNED_DIR / "config.yml")
shutil.copy2(RUN_DIR / "run.json", FINETUNED_DIR / "run.json")

fine_sha = hashlib.sha256((FINETUNED_DIR / "model.pth").read_bytes()).hexdigest()
fine_provenance = {
    "kind": "user_finetuned_vietocr",
    "initialized_from": base_provenance,
    "dataset": prep["dataset"],
    "dataset_revision": prep["revision"],
    "code_commit": git_commit,
    "seed": CFG["seed"],
    "selected_by": "minimum validation CER among optimizer-updated epochs",
    "source": run_meta["best_finetuned_source"],
    "sha256": fine_sha,
}
(FINETUNED_DIR / "provenance.json").write_text(json.dumps(fine_provenance, ensure_ascii=False, indent=2), encoding="utf-8")
display(pd.DataFrame([
    {"stage": "Before", "model": "VietOCR pretrained", "sha256": base_sha},
    {"stage": "After", "model": f"Our 5CD fine-tune ({run_meta['best_finetuned_source']})", "sha256": fine_sha},
]))
"""),
markdown(r"""## 8. Đánh giá cuối trên official test — cùng protocol

Đây là lần đầu official test được dùng để tính model metrics. Hai lệnh dùng đúng cùng `test.jsonl`. Notebook kiểm tra hash manifest, số mẫu và cờ `is_subset` trước khi cho phép so sánh.
"""),
code(r"""TEST_MANIFEST = DATA_DIR / "test.jsonl"
BASE_EVAL = REPORT_DIR / "pretrained-test"
FINE_EVAL = REPORT_DIR / "finetuned-test"

for model_dir, output_dir in [(BASE_DIR, BASE_EVAL), (FINETUNED_DIR, FINE_EVAL)]:
    if not (output_dir / "metrics.json").exists():
        shell(sys.executable, "-m", "training.evaluate",
              "--model", model_dir,
              "--manifest", TEST_MANIFEST,
              "--output", output_dir,
              "--device", DEVICE)

base_metrics = json.loads((BASE_EVAL / "metrics.json").read_text(encoding="utf-8"))
fine_metrics = json.loads((FINE_EVAL / "metrics.json").read_text(encoding="utf-8"))

assert base_metrics["manifest_sha256"] == fine_metrics["manifest_sha256"]
assert not base_metrics["is_subset"] and not fine_metrics["is_subset"]
assert base_metrics["samples"] == fine_metrics["samples"] == len(rows["test"])
assert base_metrics["model_sha256"] == base_sha
assert fine_metrics["model_sha256"] == fine_sha

display(pd.DataFrame([
    {"model": "VietOCR pretrained (before)", **{k: base_metrics[k] for k in ("samples", "cer", "wer_whitespace", "exact_match", "latency_p50", "latency_p95")}},
    {"model": "Our VietOCR fine-tuned on 5CD (after)", **{k: fine_metrics[k] for k in ("samples", "cer", "wer_whitespace", "exact_match", "latency_p50", "latency_p95")}},
]).style.format({"cer": "{:.2%}", "wer_whitespace": "{:.2%}", "exact_match": "{:.2%}", "latency_p50": "{:.4f}", "latency_p95": "{:.4f}"}))
"""),
markdown(r"""### 8.1 Paired bootstrap 95% confidence intervals

Vì hai model dự đoán cùng các dòng test, bootstrap lấy cùng chỉ số mẫu cho cả hai. Delta được định nghĩa `fine-tuned − pretrained`: delta âm là tốt cho CER/WER, delta dương là tốt cho Exact Match.
"""),
code(r"""from training.metrics import clean, distance

base_pred = read_jsonl(BASE_EVAL / "predictions.jsonl")
fine_pred = read_jsonl(FINE_EVAL / "predictions.jsonl")
assert [r["id"] for r in base_pred] == [r["id"] for r in fine_pred]
assert [r["truth"] for r in base_pred] == [r["truth"] for r in fine_pred]

def line_contributions(predictions):
    records = []
    for row in predictions:
        truth, pred = clean(row["truth"]), clean(row["prediction"])
        records.append({
            "char_edits": distance(truth, pred), "chars": max(1, len(truth)),
            "word_edits": distance(truth.split(), pred.split()), "words": max(1, len(truth.split())),
            "exact": float(truth == pred),
        })
    return pd.DataFrame(records)

base_parts = line_contributions(base_pred)
fine_parts = line_contributions(fine_pred)
rng_boot = np.random.default_rng(CFG["seed"] + 2026)
n = len(base_parts)
indices = rng_boot.integers(0, n, size=(CFG["bootstrap_repeats"], n))

def sampled_metrics(parts):
    values = {}
    values["CER"] = parts["char_edits"].to_numpy()[indices].sum(1) / parts["chars"].to_numpy()[indices].sum(1)
    values["WER"] = parts["word_edits"].to_numpy()[indices].sum(1) / parts["words"].to_numpy()[indices].sum(1)
    values["Exact Match"] = parts["exact"].to_numpy()[indices].mean(1)
    return values

base_boot, fine_boot = sampled_metrics(base_parts), sampled_metrics(fine_parts)
point_base = {"CER": base_metrics["cer"], "WER": base_metrics["wer_whitespace"], "Exact Match": base_metrics["exact_match"]}
point_fine = {"CER": fine_metrics["cer"], "WER": fine_metrics["wer_whitespace"], "Exact Match": fine_metrics["exact_match"]}

comparison_rows = []
for metric in ("CER", "WER", "Exact Match"):
    delta = fine_boot[metric] - base_boot[metric]
    comparison_rows.append({
        "metric": metric,
        "pretrained": point_base[metric],
        "pretrained_ci_low": np.quantile(base_boot[metric], .025),
        "pretrained_ci_high": np.quantile(base_boot[metric], .975),
        "fine_tuned": point_fine[metric],
        "fine_tuned_ci_low": np.quantile(fine_boot[metric], .025),
        "fine_tuned_ci_high": np.quantile(fine_boot[metric], .975),
        "delta_after_minus_before": point_fine[metric] - point_base[metric],
        "delta_ci_low": np.quantile(delta, .025),
        "delta_ci_high": np.quantile(delta, .975),
    })
comparison = pd.DataFrame(comparison_rows)
display(comparison.style.format({c: "{:.2%}" for c in comparison.columns if c != "metric"}))
"""),
code(r"""# Separate lower-is-better metrics from higher-is-better Exact Match.
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [2, 1]})
colors = ["#2563eb", "#f97316"]
labels = ["VietOCR pretrained\n(before)", "Our 5CD fine-tune\n(after)"]

for metric_index, metric in enumerate(["CER", "WER"]):
    row = comparison.set_index("metric").loc[metric]
    values = np.array([row["pretrained"], row["fine_tuned"]]) * 100
    lower = np.array([row["pretrained_ci_low"], row["fine_tuned_ci_low"]]) * 100
    upper = np.array([row["pretrained_ci_high"], row["fine_tuned_ci_high"]]) * 100
    x = np.array([metric_index - .18, metric_index + .18])
    axes[0].bar(x, values, width=.34, color=colors, edgecolor="white")
    axes[0].errorbar(x, values, yerr=np.vstack([values-lower, upper-values]), fmt="none", ecolor="#111827", capsize=4)
    for xi, value in zip(x, values): axes[0].text(xi, value, f"{value:.2f}%", ha="center", va="bottom", fontsize=9)
axes[0].set_xticks([0, 1], ["CER ↓", "Whitespace-WER ↓"])
axes[0].set_ylabel("Percent")
axes[0].set_title("Error rates — lower is better")
axes[0].legend(handles=[plt.Rectangle((0,0),1,1,color=c) for c in colors], labels=labels, frameon=False)

row = comparison.set_index("metric").loc["Exact Match"]
values = np.array([row["pretrained"], row["fine_tuned"]]) * 100
lower = np.array([row["pretrained_ci_low"], row["fine_tuned_ci_low"]]) * 100
upper = np.array([row["pretrained_ci_high"], row["fine_tuned_ci_high"]]) * 100
x = np.arange(2)
axes[1].bar(x, values, color=colors, width=.62)
axes[1].errorbar(x, values, yerr=np.vstack([values-lower, upper-values]), fmt="none", ecolor="#111827", capsize=4)
axes[1].set_xticks(x, ["Before", "After"])
axes[1].set_ylabel("Percent")
axes[1].set_title("Exact Match — higher is better")
for xi, value in zip(x, values): axes[1].text(xi, value, f"{value:.2f}%", ha="center", va="bottom", fontsize=9)

fig.suptitle("VietOCR before vs our VietOCR fine-tuned on 5CD\nOfficial held-out test · paired bootstrap 95% CI", fontsize=15, fontweight="bold")
fig.tight_layout()
fig.savefig(PLOT_DIR / "pretrained-vs-our-finetuned.png", dpi=200, bbox_inches="tight")
plt.show()
"""),
code(r"""cer_delta = fine_metrics["cer"] - base_metrics["cer"]
wer_delta = fine_metrics["wer_whitespace"] - base_metrics["wer_whitespace"]
exact_delta = fine_metrics["exact_match"] - base_metrics["exact_match"]
if cer_delta < 0:
    verdict = "Fine-tuning improved held-out CER."
else:
    verdict = "The selected fine-tuned epoch did not outperform pretrained VietOCR on held-out CER."

print(verdict)
print(f"CER: {base_metrics['cer']:.2%} → {fine_metrics['cer']:.2%} (Δ {cer_delta:+.2%})")
print(f"WER: {base_metrics['wer_whitespace']:.2%} → {fine_metrics['wer_whitespace']:.2%} (Δ {wer_delta:+.2%})")
print(f"Exact Match: {base_metrics['exact_match']:.2%} → {fine_metrics['exact_match']:.2%} (Δ {exact_delta:+.2%})")
print("Deployment selection:", run_meta["best_source"])
"""),
markdown(r"""## 9. Error analysis

Không dừng ở một con số tổng. Ta xem phân phối lỗi theo dòng, quan hệ với độ dài nhãn, mẫu tốt/xấu và các phép thế/xóa/chèn ký tự phổ biến. Đây là dữ liệu để đặt giả thuyết cho run tiếp theo; không dùng test để chọn lại checkpoint của run hiện tại.
"""),
code(r"""analysis = pd.DataFrame({
    "id": [r["id"] for r in fine_pred],
    "truth": [r["truth"] for r in fine_pred],
    "base_prediction": [r["prediction"] for r in base_pred],
    "fine_prediction": [r["prediction"] for r in fine_pred],
    "base_line_cer": base_parts["char_edits"] / base_parts["chars"],
    "fine_line_cer": fine_parts["char_edits"] / fine_parts["chars"],
})
analysis["characters"] = analysis["truth"].map(len)
analysis["cer_delta"] = analysis["fine_line_cer"] - analysis["base_line_cer"]

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
sns.histplot(analysis["base_line_cer"], bins=40, stat="density", element="step", fill=False, label="Before", ax=axes[0])
sns.histplot(analysis["fine_line_cer"], bins=40, stat="density", element="step", fill=False, label="After", ax=axes[0])
axes[0].set(title="Per-line CER distribution", xlabel="Line CER"); axes[0].legend()

sns.scatterplot(data=analysis, x="characters", y="fine_line_cer", alpha=.35, s=24, ax=axes[1])
axes[1].set(title="Fine-tuned CER vs label length", xlabel="Reference characters", ylabel="Line CER")

limit = max(1.0, np.quantile(np.r_[analysis["base_line_cer"], analysis["fine_line_cer"]], .98))
axes[2].scatter(analysis["base_line_cer"], analysis["fine_line_cer"], alpha=.35, s=22, color="#475569")
axes[2].plot([0, limit], [0, limit], "--", color="#dc2626", label="equal")
axes[2].set(xlim=(0, limit), ylim=(0, limit), xlabel="Before line CER", ylabel="After line CER", title="Paired line errors")
axes[2].legend()

fig.tight_layout()
fig.savefig(PLOT_DIR / "error-analysis.png", dpi=180, bbox_inches="tight")
plt.show()
"""),
code(r"""from collections import Counter

def edit_operations(reference, hypothesis):
    a, b = list(reference), list(hypothesis)
    dp = np.zeros((len(a)+1, len(b)+1), dtype=np.int32)
    dp[:, 0] = np.arange(len(a)+1); dp[0, :] = np.arange(len(b)+1)
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            dp[i, j] = min(dp[i-1, j]+1, dp[i, j-1]+1, dp[i-1, j-1]+(a[i-1] != b[j-1]))
    operations = []
    i, j = len(a), len(b)
    while i or j:
        if i and j and dp[i, j] == dp[i-1, j-1] + (a[i-1] != b[j-1]):
            if a[i-1] != b[j-1]: operations.append(("substitution", a[i-1], b[j-1]))
            i -= 1; j -= 1
        elif i and dp[i, j] == dp[i-1, j] + 1:
            operations.append(("deletion", a[i-1], "∅")); i -= 1
        else:
            operations.append(("insertion", "∅", b[j-1])); j -= 1
    return operations

error_counter = Counter()
for row in fine_pred:
    error_counter.update(edit_operations(clean(row["truth"]), clean(row["prediction"])))

top_errors = pd.DataFrame([
    {"operation": op, "reference": ref, "prediction": pred, "count": count}
    for (op, ref, pred), count in error_counter.most_common(25)
])
display(top_errors)

fig, ax = plt.subplots(figsize=(11, 7))
plot_errors = top_errors.head(15).copy()
plot_errors["label"] = plot_errors.apply(lambda r: f"{r['operation']}: {r['reference']} → {r['prediction']}", axis=1)
sns.barplot(data=plot_errors, y="label", x="count", color="#f97316", ax=ax)
ax.set(title="Top character edit patterns — fine-tuned model", xlabel="Count", ylabel="")
fig.tight_layout()
fig.savefig(PLOT_DIR / "top-character-errors.png", dpi=180, bbox_inches="tight")
plt.show()
"""),
code(r"""test_by_id = {r["id"]: r for r in rows["test"]}
worst = analysis.sort_values("fine_line_cer", ascending=False).head(8)
fig, axes = plt.subplots(4, 2, figsize=(16, 12))
for ax, (_, item) in zip(axes.flat, worst.iterrows()):
    source = test_by_id[item["id"]]
    with Image.open(DATA_DIR / source["image"]) as image:
        ax.imshow(image.convert("RGB"))
    ax.set_title(
        f"CER={item['fine_line_cer']:.2f}\nGT: {item['truth'][:90]}\nPR: {item['fine_prediction'][:90]}",
        fontsize=8, loc="left"
    )
    ax.axis("off")
fig.suptitle("Highest-CER test examples — qualitative audit, not model selection", fontweight="bold")
fig.tight_layout()
fig.savefig(PLOT_DIR / "worst-test-examples.png", dpi=180, bbox_inches="tight")
plt.show()

display(analysis.nsmallest(10, "cer_delta")[["truth", "base_prediction", "fine_prediction", "cer_delta"]])
display(analysis.nlargest(10, "cer_delta")[["truth", "base_prediction", "fine_prediction", "cer_delta"]])
"""),
markdown(r"""## 10. Cách diễn giải kết quả

- Nếu fine-tuned CER/WER giảm và Exact Match tăng, có thể nói run này cải thiện trên official held-out test — kèm số liệu và CI.
- Nếu metric trái chiều, báo cáo trade-off thay vì gọi chung là “accuracy tăng”.
- Nếu fine-tuned kém hơn, đây vẫn là kết quả nghiên cứu hợp lệ. Kiểm tra training curves, vocabulary audit, learning rate, domain/style và error patterns.
- Chỉ dùng validation để thiết kế run tiếp theo. Khi đã đổi hyperparameter dựa trên test, test không còn là đánh giá mù cho chuỗi thí nghiệm đó.

Các ablation hợp lý cho **run khác**: learning rate `1e-5` so với `3e-5`; vocabulary gốc so với vocabulary mở rộng đã audit; freeze CNN ngắn hạn so với full-model fine-tune. Mỗi lần chỉ đổi một yếu tố và dùng run name mới.
"""),
markdown(r"""## 11. Đóng gói model, report và provenance

Bundle `handwriting-model.zip` chứa checkpoint **đã fine-tune** tốt nhất cho mục đích nghiên cứu. Nếu validation cho thấy pretrained tốt hơn, app production nên dùng deployment-selected `model.pth` của run; notebook vẫn ghi rõ kết luận thay vì đổi nhãn bundle.
"""),
code(r"""REPORT_DIR.mkdir(parents=True, exist_ok=True)
comparison_payload = {
    "experiment": "VietOCR pretrained vs our VietOCR fine-tuned on 5CD",
    "verdict": verdict,
    "config": CFG,
    "code_commit": git_commit,
    "dataset": {"name": prep["dataset"], "revision": prep["revision"], "counts": prep["counts"]},
    "before": {"provenance": base_provenance, "metrics": base_metrics},
    "after": {"provenance": fine_provenance, "metrics": fine_metrics},
    "paired_bootstrap_95_ci": comparison.to_dict(orient="records"),
    "selection": {
        "best_finetuned_source": run_meta["best_finetuned_source"],
        "deployment_source": run_meta["best_source"],
        "fine_tuned_improved_validation": run_meta["fine_tuned_improved"],
    },
}
(REPORT_DIR / "comparison.json").write_text(json.dumps(comparison_payload, ensure_ascii=False, indent=2), encoding="utf-8")
top_errors.to_csv(REPORT_DIR / "top-character-errors.csv", index=False)
analysis.to_csv(REPORT_DIR / "paired-error-analysis.csv", index=False)
shutil.copy2(DATA_DIR / "preparation.json", REPORT_DIR / "data-preparation.json")
shutil.copy2(RUN_DIR / "run.json", REPORT_DIR / "training-run.json")
shutil.copy2(RUN_DIR / "metrics.jsonl", REPORT_DIR / "training-history.jsonl")

MODEL_PACKAGE = ROOT / "handwriting-model"
if MODEL_PACKAGE.exists(): shutil.rmtree(MODEL_PACKAGE)
MODEL_PACKAGE.mkdir()
for source, name in [
    (FINETUNED_DIR / "model.pth", "model.pth"),
    (FINETUNED_DIR / "config.yml", "config.yml"),
    (FINETUNED_DIR / "run.json", "run.json"),
    (FINETUNED_DIR / "provenance.json", "provenance.json"),
    (REPORT_DIR / "comparison.json", "comparison.json"),
]:
    shutil.copy2(source, MODEL_PACKAGE / name)

model_zip = Path(shutil.make_archive(str(ROOT / "handwriting-model"), "zip", ROOT, MODEL_PACKAGE.name))
report_zip = Path(shutil.make_archive(str(ROOT / f"{CFG['run_name']}-report"), "zip", REPORT_DIR.parent, REPORT_DIR.name))

artifact_rows = []
for artifact in (model_zip, report_zip, PLOT_DIR / "pretrained-vs-our-finetuned.png"):
    artifact_rows.append({
        "file": artifact.name,
        "bytes": artifact.stat().st_size,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    })
artifact_manifest = pd.DataFrame(artifact_rows)
artifact_manifest.to_csv(REPORT_DIR / "artifact-manifest.csv", index=False)
display(artifact_manifest)
"""),
code(r"""display(FileLink(str(model_zip)))
display(FileLink(str(report_zip)))
display(FileLink(str(PLOT_DIR / "pretrained-vs-our-finetuned.png")))
display(FileLink(str(REPORT_DIR / "comparison.json")))
"""),
markdown(r"""## Tài liệu phương pháp

- [VietOCR — repository chính thức](https://github.com/pbcquoc/vietocr): kiến trúc Transformer OCR, format annotation và pretrained model.
- [5CD Vietnamese Handwriting OCR v2](https://huggingface.co/datasets/5CD-AI/Viet-Handwriting-OCR-v2): dataset card, điều kiện truy cập và thông tin phiên bản dữ liệu.
- [PyTorch Automatic Mixed Precision](https://docs.pytorch.org/docs/stable/notes/amp_examples.html): `autocast` và gradient scaling.
- [PyTorch Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html): seed, deterministic algorithms và giới hạn tái lập giữa môi trường.

Các lựa chọn learning rate, patience và augmentation trong notebook là giả thuyết thực nghiệm của project; chúng không được trình bày như thông số tối ưu do nguồn chính thức bảo đảm.
"""),
markdown(r"""## Checklist trước khi đưa lên GitHub/CV

- [ ] Notebook chạy từ đầu đến cuối trên một Kaggle version sạch.
- [ ] Không có token, dataset images hoặc checkpoint lớn trong Git.
- [ ] Hai model có SHA-256 và cùng test manifest hash.
- [ ] README dùng số sinh ra từ `comparison.json`, không chép số của run cũ.
- [ ] Biểu đồ ghi rõ hướng tốt/xấu của từng metric.
- [ ] Nếu fine-tuning không cải thiện, ghi đúng là negative result và nêu bước ablation tiếp theo.
- [ ] Dataset/model/license được dẫn nguồn; bundle weights không được phát hành nếu điều khoản không cho phép.

Một câu CV trung thực có thể dùng sau khi có kết quả:

> Built a reproducible VietOCR fine-tuning and paired evaluation pipeline on the gated 5CD Vietnamese handwriting dataset, with leakage checks, checkpoint provenance, CER/WER/Exact Match, bootstrap confidence intervals, and qualitative error analysis.
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

destination = Path("notebooks/01_kaggle_finetune.ipynb")
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(destination, "cells=", len(cells))
