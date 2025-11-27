import argparse
import random
from openai import OpenAI
import torch.cuda.nvtx as nvtx

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
    "What are the benefits of exercise?",
    "How does a car engine work?",
    "What is the history of the internet?",
    "Describe your ideal vacation.",
    "What are the biggest challenges facing humanity today?",
    "Explain quantum entanglement.",
    "If you could have any superpower, what would it be and why?",
    "Discuss the ethical implications of artificial intelligence.",
    "What is the most beautiful place you've ever imagined?",
    "Write a short poem about the ocean.",
    "Compare and contrast democracy and communism.",
    "What is the process of photosynthesis?",
    "Describe a typical day in the life of a software engineer.",
    "What is your favorite book and why?",
    "How do meteorologists predict the weather?",
    "What are the different types of renewable energy?",
    "Discuss the impact of social media on society.",
    "What is the best way to learn a new language?",
    "Explain the concept of supply and demand.",
    "Write a dialogue between two friends planning a trip.",
    "What are the major theories of psychology?",
    "Describe the architecture of the Roman Colosseum.",
    "What role does art play in culture?",
    "How do vaccines work to prevent disease?",
    "What is the difference between augmented reality and virtual reality?",
    "Discuss the importance of critical thinking.",
    "What are the steps involved in baking a cake?",
    "Explain the concept of a black hole.",
    "Write a persuasive essay about the importance of environmental conservation.",
    "What are the key elements of a good story?",
    "How does blockchain technology work?",
    "Describe the life cycle of a butterfly.",
    "What is the significance of the Mona Lisa?",
    "Discuss the challenges and opportunities of remote work.",
    "What are the basic principles of healthy eating?",
    "Explain the theory of evolution by natural selection.",
    "Write a script for a short commercial.",
    "What are the benefits of meditation?",
    "Describe the history of space exploration.",
    "What is the purpose of education?",
]


def main(args):
    # Point the OpenAI client at the local vLLM server.
    client = OpenAI(api_key=args.api_key, base_url=args.api_base)

    if args.num_prompts > 1:
        prompts = random.choices(PROMPTS, k=args.num_prompts)
    else:
        prompts = [args.prompt]

    nvtx.range_push("Benchmark Run")  # Start of the overall benchmark range
    for i, prompt in enumerate(prompts):
        nvtx.range_push(f"Prompt {i+1}")  # Start range for a single prompt
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
        nvtx.range_pop()  # End range for a single prompt
    nvtx.range_pop()  # End of the overall benchmark range


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
                        default="Qwen/Qwen2.5-3B-Instruct",              ###########WE CHANGE THIS HERE!!!!!
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
