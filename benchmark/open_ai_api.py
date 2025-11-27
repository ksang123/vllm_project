import argparse
import logging
from openai import OpenAI
import torch.cuda.nvtx as nvtx
from config_handler import load_config
from prompt_generator import get_prompts

def main(args):
    logger = logging.getLogger("vllm_benchmark")
    # Point the OpenAI client at the local vLLM server.
    client = OpenAI(api_key=args.api_key, base_url=args.api_base)

    # Get prompts from the generator
    prompts = get_prompts(
        dataset_name=args.dataset_name,
        num_prompts=args.num_prompts,
        prompt_len=args.random_input_len,
        seed=args.seed
    )

    if not prompts:
        logger.error(f"Could not generate prompts for dataset '{args.dataset_name}'. Please check the dataset name and prompt generator implementation.")
        return

    nvtx.range_push("Benchmark Run")  # Start of the overall benchmark range
    for i, prompt in enumerate(prompts):
        nvtx.range_push(f"Prompt {i+1}")  # Start range for a single prompt
        logger.info(f"\n--- Prompt {i+1}/{len(prompts)}: '{prompt[:50]}...' ---")
        
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

        try:
            completion = client.completions.create(**filtered_args)

            logger.info("\n=== Completion ===")
            for choice in completion.choices:
                logger.info(f"[choice {choice.index}] finish={choice.finish_reason}")
                logger.info(choice.text.strip() + "\n")

            if completion.usage:
                logger.info(
                    f"Tokens — prompt: {completion.usage.prompt_tokens}, "
                    f"completion: {completion.usage.completion_tokens}, "
                    f"total: {completion.usage.total_tokens}")
        except Exception as e:
            logger.error(f"An error occurred during API call: {e}")

        nvtx.range_pop()  # End range for a single prompt
    nvtx.range_pop()  # End of the overall benchmark range


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A simple script to benchmark a vLLM server based on a config file.")

    # Arguments from benchmark_config
    parser.add_argument("--api-base", type=str, help="API base URL for the vLLM server.")
    parser.add_argument("--api-key", type=str, help="API key for the vLLM server.")
    parser.add_argument("--model", type=str, default=None, help="The model to use for the benchmark.")
    parser.add_argument("--dataset-name", type=str, help='Dataset to use ("random", "sharegpt", etc.)')
    parser.add_argument("--num-prompts", type=int, help="Total number of prompts to send during the test")
    parser.add_argument("--max-concurrency", type=int, help="Maximum number of concurrent requests")
    parser.add_argument("--request-rate", type=str, help='Target request generation rate (requests per second, "inf" for max throughput)')
    parser.add_argument("--random-input-len", type=int, help="Length (in tokens) of each generated input prompt (if dataset_name is 'random')")
    parser.add_argument("--random-output-len", type=int, help="Desired length (in tokens) of the model's output (if dataset_name is 'random')")
    parser.add_argument("--temperature", type=float, help="Sampling temperature for generation")
    parser.add_argument("--top-p", type=float, help="Top-p sampling parameter")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--ignore-eos", action="store_true", help="If true, generation will not stop at EOS token until max_output_len is reached")
    parser.add_argument("--n", type=int, help="Number of output sequences to return for the given prompt")
    parser.add_argument("--best-of", type=int, help="Number of output sequences that are generated from the prompt")
    parser.add_argument("--presence-penalty", type=float, help="Penalty for new tokens already in the prompt")
    parser.add_argument("--frequency-penalty", type=float, help="Penalty for new tokens based on their frequency in the prompt")
    parser.add_argument("--top-k", type=int, help="Number of highest probability vocabulary tokens to keep for top-k-filtering")
    parser.add_argument("--use-beam-search", action="store_true", help="Whether to use beam search instead of sampling")
    parser.add_argument("--stop", nargs='*', help="A list of strings that will stop the generation")
    parser.add_argument("--stream", action="store_true", help="Whether to stream the output")
    parser.add_argument("--logprobs", type=int, help="Number of log probabilities to return")

    args = parser.parse_args()
    
    # Load config to get default values if they are not provided in args
    config = load_config()
    
    if args.model is None:
        if config and 'server_config' in config and 'model' in config['server_config']:
            args.model = config['server_config']['model']
        else:
            # Fallback if config is not available or model is not specified
            logging.getLogger("vllm_benchmark").warning(" --model argument not provided and could not determine from config. Using a default model.")
            args.model = "meta-llama/Llama-2-7b-hf"

    if args.api_base is None:
        if config and 'benchmark_config' in config and 'api_base' in config['benchmark_config']:
            args.api_base = config['benchmark_config']['api_base']
        else:
            args.api_base = "http://localhost:8000/v1"

    if args.api_key is None:
        if config and 'server_config' in config and 'api_key' in config['server_config']:
            args.api_key = config['server_config']['api_key']
        else:
            args.api_key = "EMPTY"
            
    main(args)
