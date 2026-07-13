# Text Generation with GPT-2

**Internship Task – Prodigy InfoTech (Generative AI)**

## Overview

This project fine-tunes GPT-2, a transformer-based language model developed by OpenAI, on a custom text dataset to generate coherent, contextually relevant text that mimics the style and structure of the training data.

## Project Structure

```
├── dataset/
│   └── dataset.txt          # Custom training data (samples separated by <|endoftext|>)

├── train.py                 # Fine-tuning script (Dataset -> Tokenizer -> Model -> Trainer -> Save)

├── generate.py               # Script to load the fine-tuned model and generate text interactively

├── requirements.txt
└── README.md
```

> Note: The `output/` folder (trained model weights and checkpoints) is not included in this
> repository due to file size. Run `train.py` to reproduce it locally.

## How It Works

1. **Dataset** – Custom text samples are loaded and split on the `<|endoftext|>` delimiter.
2. **Tokenizer** – GPT-2's tokenizer (`AutoTokenizer`) processes the text, with the EOS token used as the padding token.
3. **Model** – A pretrained `gpt2` model (`AutoModelForCausalLM`) is loaded and fine-tuned.
4. **Training** – Hugging Face's `Trainer` API handles the fine-tuning loop (5 epochs, batch size 2, learning rate 5e-5, fp16 mixed precision on GPU).
5. **Generation** – The fine-tuned model generates text continuations from user-provided prompts using top-k / top-p sampling.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Usage

**1. Fine-tune the model**

```bash
python train.py
```

This trains GPT-2 on `dataset/dataset.txt` and saves the fine-tuned model to `output/final_model`.

**2. Generate text**

```bash
python generate.py
```

Enter a prompt when asked, and the model will generate a continuation in the style of the training data. Type `exit` to quit.

## Sample Output

**Prompt:** `What is Artificial Intelligence?`

> AI enables intelligent systems to perform tasks that humans cannot. It improves decision-making, adapts text generation techniques and improves human productivity through algorithmic reasoning. Examples of this include medical assistants who analyze patient data for disease risk or treatment recommendations automatically from clinical records...

**Prompt:** `What is transfer learning in Transformer models?`

> Transfer Learning uses machine translation to automatically learn relationships between words, making it highly efficient for language model developers. During fine-tuning of a neural network with contextually relevant text sources, contextual information may become available while retaining the original content or patterns that were previously hidden during training...

## Training Details

| Parameter        | Value                 |
| ---------------- | --------------------- |
| Base model       | GPT-2 (small)         |
| Training samples | 190                   |
| Epochs           | 5                     |
| Batch size       | 2                     |
| Learning rate    | 5e-5                  |
| Precision        | fp16 (GPU)            |
| Hardware         | NVIDIA RTX 3050 (6GB) |
| Final train loss | ~1.68                 |

## Notes & Limitations

- Trained on a small custom dataset (190 samples), so outputs are style-matched but not always fully coherent over long generations.
- Not intended for factual accuracy — this is a style/pattern-generation model, not a knowledge base.
- Future improvements: larger dataset, custom training loop with gradient accumulation and LR scheduling, longer training with overfitting checks.

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
