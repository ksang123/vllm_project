# Prompt generator for the benchmark
# This file contains the prompt generator for the benchmark
# It is used to generate prompts for the benchmark based on the configuration.

import random
import string

SHAREGPT_PROMPTS = [
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

def _generate_random_string_prompt(length: int) -> str:
    """Generates a random prompt of a given length from letters, digits, and spaces."""
    return "".join(random.choices(string.ascii_letters + string.digits + " ", k=length))

def get_prompts(dataset_name: str, num_prompts: int, prompt_len: int = 512, seed: int = 42) -> list[str]:
    """
    Generates a list of prompts based on the specified dataset and parameters.

    Args:
        dataset_name: The name of the dataset to use ("random", "sharegpt").
        num_prompts: The number of prompts to generate.
        prompt_len: The length of the prompt to generate if dataset_name is "random".
        seed: The random seed for reproducibility.

    Returns:
        A list of prompts.
    """
    random.seed(seed)
    
    if dataset_name == "random":
        return [_generate_random_string_prompt(prompt_len) for _ in range(num_prompts)]
    elif dataset_name == "sharegpt":
        # Ensure we don't request more prompts than are available if we shouldn't have duplicates
        if num_prompts > len(SHAREGPT_PROMPTS):
            # If more prompts are requested than available, sample with replacement
            return random.choices(SHAREGPT_PROMPTS, k=num_prompts)
        else:
            # Otherwise, sample without replacement
            return random.sample(SHAREGPT_PROMPTS, k=num_prompts)
    else:
        # TODO: Add support for loading other datasets from Hugging Face
        # For now, return an empty list if the dataset is not recognized.
        return []

def get_random_prompt(dataset_name: str = "sharegpt") -> str:
    """Returns a single random prompt from the specified dataset."""
    if dataset_name == "sharegpt":
        return random.choice(SHAREGPT_PROMPTS)
    # This function is less meaningful for the "random" dataset,
    # but we can generate one with a default length.
    elif dataset_name == "random":
        return _generate_random_string_prompt(512)
    return ""