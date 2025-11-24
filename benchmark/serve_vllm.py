import os
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

model_name = "Qwen/Qwen2.5-3B-Instruct"
tp_size = int(os.getenv("VLLM_TENSOR_PARALLEL", torch.cuda.device_count() or 1))

llm = LLM(
    model=model_name,
    tensor_parallel_size=tp_size,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
messages_list = [
    [{"role": "user", "content": prompt}]
    for prompt in prompts
]
texts = tokenizer.apply_chat_template(
    messages_list,
    tokenize=False,
    add_generation_prompt=True,
)

outputs = llm.generate(texts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

outputs = llm.chat(messages_list, sampling_params)
for idx, output in enumerate(outputs):
    prompt = prompts[idx]
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
