from transformers import pipeline
import torch
from huggingface_hub import login

# Replace with your actual token
login(token="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")


# Check if CUDA (GPU) is available
device = 0 if torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU
print(device)

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct", device=device)
pipe(messages)