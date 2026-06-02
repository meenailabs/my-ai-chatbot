import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import sys
import os

# Disable everything
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging = __import__("logging")
logging.getLogger("transformers").setLevel(logging.CRITICAL)

# Also suppress stdout during loading
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        return self
    def __exit__(self, *args):
        sys.stdout.close()
        sys.stdout = self._original_stdout

print("Loading AI...", flush=True)

with SuppressOutput():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    tokenizer.pad_token = tokenizer.eos_token

print("AI is ready! Start chatting.\n", flush=True)

chat_history_ids = None

while True:
    user = input("You: ")
    if user.lower() in ["quit", "exit", "q"]:
        print("Bye!")
        break
    
    if not user.strip():
        continue
    
    new_input_ids = tokenizer.encode(user + tokenizer.eos_token, return_tensors="pt")
    attention_mask = torch.ones_like(new_input_ids)
    
    if chat_history_ids is not None:
        chat_history_ids = chat_history_ids[:, -512:]
        attention_old = torch.ones_like(chat_history_ids)
        input_ids = torch.cat([chat_history_ids, new_input_ids], dim=1)
        attention_mask = torch.cat([attention_old, attention_mask], dim=1)
    else:
        input_ids = new_input_ids
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
        )
    
    new_response = output_ids[0][input_ids.shape[1]:]
    response = tokenizer.decode(new_response, skip_special_tokens=True)
    
    print(f"Bot: {response}\n")
    chat_history_ids = output_ids