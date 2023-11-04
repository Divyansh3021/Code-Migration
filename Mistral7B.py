# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
device = "cpu"
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

prompt = """[INST]Convert this python code in C++

for i in range(23):
    if num%2 == 0:
        print("hello")

[/INST]"""

model_inputs = tokenizer([prompt], return_tensors = "pt").to(device)

model.to(device)

generated_ids = model.generate(**model_inputs, max_new_tokens = 100, do_sample = True)

tokenizer.batch_decode(generated_ids[0])