import sys
from pathlib import Path
import torch
from transformers import DistilBertConfig, DistilBertForSequenceClassification, PreTrainedTokenizerFast
from safetensors.torch import load_file as load_safetensors

# Add project root to sys.path so ingestion and classifier packages import correctly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
MODEL_DIR = PROJECT_ROOT / 'classifier' / 'checkpoint'

# Load tokenizer from local tokenizer.json
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=str(MODEL_DIR / 'tokenizer.json')
)

# Manually load model config and weights to avoid HF hub logic
config = DistilBertConfig.from_json_file(str(MODEL_DIR / 'config.json'))
model = DistilBertForSequenceClassification(config)

# Load safetensors weights
state_dict = load_safetensors(str(MODEL_DIR / 'model.safetensors'))
model.load_state_dict(state_dict)
model.eval()

def classify_text(text: str) -> str:
    """
    Return the classifier’s predicted label for the given text.
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = logits.argmax(dim=-1).item()
    return model.config.id2label[pred_id]
