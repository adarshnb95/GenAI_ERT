# File: classifier/predict.py

import os
import sys
from pathlib import Path
import torch
from transformers import DistilBertConfig, DistilBertForSequenceClassification, PreTrainedTokenizerFast
from safetensors.torch import load_file as load_safetensors

# Determine project root so we can locate the checkpoint directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "classifier" / "checkpoint"

# We will cache these on first use
_tokenizer = None
_model = None


def _load_model_and_tokenizer():
    """
    Lazy‐load the DistilBERT model and tokenizer from MODEL_DIR.
    If anything fails, leave _tokenizer and _model as None.
    """
    global _tokenizer, _model

    # If already loaded, do nothing
    if _tokenizer is not None and _model is not None:
        return

    try:
        # Ensure the checkpoint folder exists and has tokenizer.json
        if not MODEL_DIR.exists() or not (MODEL_DIR / "tokenizer.json").is_file():
            print(">>> [classifier.predict] No checkpoint found; returning UNCLASSIFIED at inference.")
            return

        # Load the tokenizer from the saved tokenizer.json
        _tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(MODEL_DIR / "tokenizer.json")
        )

        # Load the DistilBERT configuration
        config = DistilBertConfig.from_json_file(str(MODEL_DIR / "config.json"))

        # Initialize a fresh model from that config
        _model = DistilBertForSequenceClassification(config)

        # Load the safetensors file into the model
        state_dict = load_safetensors(str(MODEL_DIR / "model.safetensors"))
        _model.load_state_dict(state_dict)
        _model.eval()

        print(">>> [classifier.predict] Model & tokenizer loaded successfully.")

    except Exception as e:
        # If anything went wrong, wipe them out
        print(f">>> [classifier.predict] Failed to load model/tokenizer: {e}")
        _tokenizer = None
        _model = None


def classify_text(text: str) -> str:
    """
    Return the classifier’s predicted label for the given text.
    If loading hasn’t occurred or fails, return "UNCLASSIFIED".
    """
    # On first call, try to load the model/tokenizer
    if _model is None or _tokenizer is None:
        _load_model_and_tokenizer()

    # If still not loaded, bail out
    if _model is None or _tokenizer is None:
        return "UNCLASSIFIED"

