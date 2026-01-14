from dotenv import load_dotenv
from huggingface_hub import login
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def next_token_distribution(model, tokenizer, prompt: str, temperature: float = 1.0):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model(**inputs)  # out.logits: [batch, seq_len, vocab]
        logits = out.logits[0, -1, :]  # logits for NEXT token after the prompt

    if temperature != 1.0:
        logits = logits / temperature

    probs = torch.softmax(logits, dim=-1)  # [vocab]
    return probs

load_dotenv()

token = os.getenv("HF_TOKEN")
login(token=token)

model_id = "Qwen/Qwen3-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()

# Example usage
prompt = "Plan a 4-day trip to Paris."
probs = next_token_distribution(model, tokenizer, prompt, temperature=0.6)

# Print top 50 next tokens
topk = 50
vals, ids = torch.topk(probs, k=topk)
for p, tid in zip(vals.tolist(), ids.tolist()):
    tok = tokenizer.convert_ids_to_tokens(tid)
    # show token safely
    print(f"{tid:>6}  {p:>9.6f}  {repr(tok)}")

# prompt = "Plan a 4-day trip to Paris."

# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# # 1. Generate the response
# output_tokens = model.generate(
#     **inputs, 
#     max_new_tokens=256, 
#     do_sample=True,
#     temperature=0.6,
#     top_p=0.9
# )

# # Decode the tokens into readable text
# # We slice [0] because the model returns a batch, and we only have 1 prompt
# response_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
# print("\n--- Human Readable Response ---")
# print(response_text)

# # 2. Get the full list of Token IDs
# # output_tokens[0] contains both your prompt IDs and the new generated IDs
# generated_ids = output_tokens[0].tolist()

# # 3. Convert those IDs into the sub-word strings (Tokens)
# token_strings = tokenizer.convert_ids_to_tokens(generated_ids)

# # 4. Print them out
# print("--- Numerical Token IDs ---")
# print(generated_ids)

# print("\n--- Token Strings ---")
# print(token_strings)