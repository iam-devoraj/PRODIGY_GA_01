import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./output/final_model"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
model.eval()

def generate(prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    while True:
        prompt = input("\nPrompt (or 'exit'): ")
        if prompt.lower() == "exit":
            break
        print("\n" + generate(prompt))