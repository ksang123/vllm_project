import argparse
import random
from openai import OpenAI

PROMPTS = [
    "Hello, I'm a language model,",
    "The capital of France is",
    "The future of AI is",
    "San Francisco is a",
    "Python is a popular programming language for",
    "The quick brown fox jumps over the lazy dog",
    "Describe the plot of the movie Inception.",
    "What is the meaning of life?",
    "Explain the theory of relativity in simple terms.",
    "Write a short story about a robot who discovers music.",
]


def main(args):
    # Point the OpenAI client at the local vLLM server.
    client = OpenAI(api_key=args.api_key, base_url=args.api_base)

    if args.num_prompts > 1:
        prompts = random.choices(PROMPTS, k=args.num_prompts)
    else:
        prompts = [args.prompt]

    for i, prompt in enumerate(prompts):
        print(f"\n--- Prompt {i+1}/{len(prompts)}: '{prompt}' ---")
        completion = client.completions.create(model=args.model,
                                                prompt=prompt,
                                                max_tokens=args.max_tokens)

        print("\n=== Completion ===")
        for choice in completion.choices:
            print(f"[choice {choice.index}] finish={choice.finish_reason}")
            print(choice.text.strip(), "\n")

        if completion.usage:
            print(
                f"Tokens — prompt: {completion.usage.prompt_tokens}, "
                f"completion: {completion.usage.completion_tokens}, "
                f"total: {completion.usage.total_tokens}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A simple script to benchmark a vLLM server.")
    parser.add_argument("--api-key",
                        type=str,
                        default="YOUR_TOKEN",
                        help="API key for the vLLM server.")
    parser.add_argument("--api-base",
                        type=str,
                        default="http://localhost:8000/v1",
                        help="API base URL for the vLLM server.")
    parser.add_argument("--model",
                        type=str,
                        default="Qwen/Qwen2.5-3B-Instruct",
                        help="The model to use for the benchmark.")
    parser.add_argument("--prompt",
                        type=str,
                        default="San Francisco is a",
                        help="The prompt to send to the model if num-prompts is 1.")
    parser.add_argument("--max-tokens",
                        type=int,
                        default=64,
                        help="The maximum number of tokens to generate.")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1,
                        help="Number of prompts to run.")
    args = parser.parse_args()
    main(args)
