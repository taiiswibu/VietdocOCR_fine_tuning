# Model artifacts

Weights are downloaded explicitly with `python scripts/download_models.py --all`.
The release ZIP contains source code and examples, not large third-party model weights.

- `base/`: external printed-text pretrained checkpoint, initialization for your fine-tuning.
- `handwriting/`: local inference bundle; replace with YOUR Kaggle export when ready.
- `easyocr/`: CRAFT detector and Latin recognizer, downloaded once.

Keep base and finetuned weights separate. Only load trusted tensor checkpoints.
