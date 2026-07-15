import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load fine-tuned model
MODEL_PATH = "./output/final_model"

device = "cuda" if torch.cuda.is_available() else "cpu"ce

print("=" * 50)
print("GPT-2 Text Generator")
print("Device:", device)
print("=" * 50)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
model.eval()


def generate(question):
    prompt = f"Question:\n{question}\n\nAnswer:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# ==============================
# Main Loop
# ==============================

while True:

    question = input("\nEnter your question (type 'exit' to quit): ").strip()

    if question.lower() == "exit":
        print("Goodbye!")
        break

    if question == "":
        print("Please enter a question.")
        continue

    answer = generate(question)

    print("\n" + "=" * 50)
    print(answer)
    print("=" * 50)