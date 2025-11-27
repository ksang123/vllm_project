"""
benchmark/serve_vllm.py

This script serves as a basic demonstration or a local unit test for
vLLM's programmatic inference capabilities. It is NOT an API server.

It directly loads a pre-trained language model using vLLM's `LLM` class
and performs text generation for a few hardcoded prompts. This allows for
quick verification of vLLM installation and basic model functionality
without the overhead of an HTTP server.

Key characteristics:
-   **Direct Inference:** Interacts with vLLM's core library directly, not via an API endpoint.
-   **Local Execution:** Model loading and inference happen within the same Python process.
-   **Hardcoded Prompts:** Uses a small, fixed set of prompts for demonstration.
-   **No External API:** Does not expose any network interface.
-   **Not a full experiment:** Designed for quick checks, not comprehensive benchmarking.
"""
import os
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# --- Configuration for this basic test ---
# Hardcoded prompts for a quick demonstration of generation
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Sampling parameters for controlling the text generation behavior
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# The model to be loaded for this test.
# This model will be loaded directly into the GPU memory when LLM is instantiated.
model_name = "Qwen/Qwen2.5-3B-Instruct"
# Determine tensor parallel size, useful for distributed inference
tp_size = int(os.getenv("VLLM_TENSOR_PARALLEL", torch.cuda.device_count() or 1))

# --- Model Loading and Inference ---
# Instantiate the vLLM LLM object.
# This loads the model directly into memory for local inference.
llm = LLM(
    model=model_name,
    tensor_parallel_size=tp_size,
)

# Initialize tokenizer (used for chat templates in this example)
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

# Perform text generation using the loaded model
print("\\n--- Direct Generation (llm.generate) ---")
outputs = llm.generate(texts, sampling_params)

# Print the results of the generation
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# Another way to perform generation, specifically for chat models
print("\\n--- Direct Generation (llm.chat) ---")
outputs = llm.chat(messages_list, sampling_params)
for idx, output in enumerate(outputs):
    prompt = prompts[idx]
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
