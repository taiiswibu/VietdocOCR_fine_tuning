# Sources and third-party terms

Project integration code is provided under MIT (LICENSE). Third-party libraries, configurations, pretrained weights, user samples and datasets retain their own terms.

- VietOCR source/configuration: https://github.com/pbcquoc/vietocr — Apache 2.0; `configs/vietocr.yml` derives from upstream base/VGG configs. Runtime 0.3.13 dependency pins differ from current GitHub main; compatible pinned dependencies are declared in pyproject.toml.
- EasyOCR: https://github.com/JaidedAI/EasyOCR — Apache 2.0. Uses detector and Latin recognition weights downloaded by its own installer.
- Base VietOCR checkpoint mirror: https://huggingface.co/doanhm/vgg_transformer . Downloader records exact revision/source; inspect original model terms before reuse beyond this research project.
- Community handwriting baseline: https://huggingface.co/vudang449/vietocr-vietnamese-handwriting . External fine-tuned model, never attributed as user training. Model card license declaration does not erase dataset conditions.
- Handwriting data: https://huggingface.co/datasets/5CD-AI/Viet-Handwriting-OCR-v2 . Publisher states CC BY-NC 4.0 and gated access. Not redistributed in this package.
- PyMuPDF: https://pymupdf.readthedocs.io/ — AGPL/commercial licensing; consider the applicable terms when redistributing/deploying.
- FastAPI: https://fastapi.tiangolo.com/ ; PyTorch: https://pytorch.org/ ; Hugging Face Datasets: https://huggingface.co/docs/datasets/ .
- Three PDF samples and reference screenshot were supplied by the user; included for private local testing, ignored by Git by default.

No third-party model weights or gated dataset records are included in the source release ZIP. Source links and revision identifiers are provided for explicit downloads.
