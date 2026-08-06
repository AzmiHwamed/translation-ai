import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_NAME = "facebook/nllb-200-distilled-600M"


device = "cuda" if torch.cuda.is_available() else "cpu"


print("Loading translation model...")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    use_safetensors=True,
)
if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
model.to(device)

# Inference-only: disables dropout etc. Always do this for a model
# that's only ever used for generation, not training.
model.eval()


print(f"Translation model loaded on {device}")
