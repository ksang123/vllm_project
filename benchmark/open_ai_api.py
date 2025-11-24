from openai import OpenAI

# Point the OpenAI client at the local vLLM server.
API_KEY = "YOUR_TOKEN"
API_BASE = "http://localhost:8000/v1"
MODEL = "Qwen/Qwen2.5-3B-Instruct"
PROMPT = "San Francisco is a"

client = OpenAI(api_key=API_KEY, base_url=API_BASE)

completion = client.completions.create(model=MODEL, prompt=PROMPT, max_tokens=64)

print("\n=== Completion ===")
for choice in completion.choices:
    print(f"[choice {choice.index}] finish={choice.finish_reason}")
    print(choice.text.strip(), "\n")

if completion.usage:
    print(
        f"Tokens — prompt: {completion.usage.prompt_tokens}, "
        f"completion: {completion.usage.completion_tokens}, "
        f"total: {completion.usage.total_tokens}"
    )
