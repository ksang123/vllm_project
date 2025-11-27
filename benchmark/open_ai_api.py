import argparse
import random
import string
from openai import OpenAI
import torch.cuda.nvtx as nvtx
from config_handler import load_config

# TODO: Implement loading datasets from Hugging Face or other sources
PROMPTS = [
    "Hello, I'm a language model,",
    "The capital of France is",
    "The future of AI is",
    "San Francisco is a",
    "Python is a popular programming language for",
    "The quick brown fox jumps over the lazy dog",
]


def generate_random_prompt(length):
    """Generates a random prompt of a given length."""
    return "".join(random.choices(string.ascii_letters + string.digits + " ", k=length))


def main(args):
    # Point the OpenAI client at the local vLLM server.
    client = OpenAI(api_key="EMPTY", base_url=args.api_base)

    if args.dataset_name == "random":
        prompts = [generate_random_prompt(args.random_input_len) for _ in range(args.num_prompts)]
    else:
        # For now, just use the hardcoded prompts for other dataset names
        prompts = random.choices(PROMPTS, k=args.num_prompts)


    nvtx.range_push("Benchmark Run")  # Start of the overall benchmark range
    for i, prompt in enumerate(prompts):
        nvtx.range_push(f"Prompt {i+1}")  # Start range for a single prompt
        print(f"\n--- Prompt {i+1}/{len(prompts)}: '{prompt[:50]}...' ---")
        
        # Filter out None values from args before passing to create
        completion_args = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": args.random_output_len,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "n": args.n,
            "best_of": args.best_of,
            "presence_penalty": args.presence_penalty,
            "frequency_penalty": args.frequency_penalty,
            "top_k": args.top_k,
            "use_beam_search": args.use_beam_search,
            "stop": args.stop if args.stop else None,
            "ignore_eos": args.ignore_eos,
            "stream": args.stream,
            "logprobs": args.logprobs,
        }
        
        filtered_args = {k: v for k, v in completion_args.items() if v is not None}

        completion = client.completions.create(**filtered_args)

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
        description="A simple script to benchmark a vLLM server based on a config file.")

    # Arguments from benchmark_config
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1", help="API base URL for the vLLM server.")
    parser.add_argument("--model", type=str, default=None, help="The model to use for the benchmark.")
    parser.add_argument("--dataset-name", type=str, default="random", help='Dataset to use ("random", "ShareGPT", etc.)')
    parser.add_argument("--num-prompts", type=int, default=10, help="Total number of prompts to send during the test")
    parser.add_argument("--max-concurrency", type=int, default=16, help="Maximum number of concurrent requests")
    parser.add_argument("--request-rate", type=str, default="inf", help='Target request generation rate (requests per second, "inf" for max throughput)')
    parser.add_argument("--random-input-len", type=int, default=512, help="Length (in tokens) of each generated input prompt (if dataset_name is 'random')")
    parser.add_argument("--random-output-len", type=int, default=128, help="Desired length (in tokens) of the model's output (if dataset_name is 'random')")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for generation")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling parameter")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--ignore-eos", action="store_true", help="If true, generation will not stop at EOS token until max_output_len is reached")
    parser.add_argument("--n", type=int, default=1, help="Number of output sequences to return for the given prompt")
    parser.add_argument("--best-of", type=int, default=1, help="Number of output sequences that are generated from the prompt")
    parser.add_argument("--presence-penalty", type=float, default=0.0, help="Penalty for new tokens already in the prompt")
    parser.add_argument("--frequency-penalty", type=float, default=0.0, help="Penalty for new tokens based on their frequency in the prompt")
    parser.add_argument("--top-k", type=int, default=-1, help="Number of highest probability vocabulary tokens to keep for top-k-filtering")
    parser.add_argument("--use-beam-search", action="store_true", help="Whether to use beam search instead of sampling")
    parser.add_argument("--stop", nargs='*', help="A list of strings that will stop the generation")
    parser.add_argument("--stream", action="store_true", help="Whether to stream the output")
    parser.add_argument("--logprobs", type=int, default=None, help="Number of log probabilities to return")

    args = parser.parse_args()
    
    # Set model from server_config if not provided
    if args.model is None:
        config = load_config()
        if config and 'server_config' in config and 'model' in config['server_config']:
            args.model = config['server_config']['model']
        else:
            # Fallback if config is not available or model is not specified
            print("Warning: --model argument not provided and could not determine from config. Using a default model.")
            args.model = "meta-llama/Llama-2-7b-hf"
            
    main(args)
