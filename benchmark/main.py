import subprocess
import sys
import argparse

# A list of common models for user convenience
SUPPORTED_MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Llama-2-7b-chat-hf",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "google/gemma-2b-it",
]


def display_menu():
    """Prints the model selection menu."""
    print("\n--- vLLM Server Launcher ---")
    print("Please choose a model to serve:")
    for i, model in enumerate(SUPPORTED_MODELS):
        print(f"  {i + 1}. {model}")
    print(f"  {len(SUPPORTED_MODELS) + 1}. Enter a custom HuggingFace model name")
    print("  0. Exit")


def get_model_choice():
    """Gets the user's model choice from the menu."""
    while True:
        try:
            choice = input(
                f"\nEnter your choice (0-{len(SUPPORTED_MODELS) + 1}): ")
            choice = int(choice)

            if 0 <= choice <= len(SUPPORTED_MODELS) + 1:
                if choice == 0:
                    return None
                if choice == len(SUPPORTED_MODELS) + 1:
                    custom_model = input(
                        "Enter the custom HuggingFace model name (e.g., 'user/model'): "
                    )
                    if custom_model:
                        return custom_model
                    else:
                        print("Error: Custom model name cannot be empty.")
                        continue
                return SUPPORTED_MODELS[choice - 1]
            else:
                print(
                    f"Invalid choice. Please enter a number between 0 and {len(SUPPORTED_MODELS) + 1}."
                )
        except ValueError:
            print("Invalid input. Please enter a number.")
        except IndexError:
            print("Invalid choice. Please select a valid option from the menu.")


def run_server(model_name, additional_args):
    """Constructs and runs the vLLM server command."""
    command = [
        sys.executable,  # Use the same python interpreter
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_name,
    ]

    # Add any additional arguments
    if additional_args:
        # The vllm entrypoint script doesn't like the '--' separator
        if additional_args and additional_args[0] == '--':
            additional_args.pop(0)
        command.extend(additional_args)

    print("\n🚀 Starting vLLM server with the following command:")
    # Format for readability
    print("  " + " \
    ".join(command))
    print("\nTo stop the server, press Ctrl+C in this terminal.")

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error starting the server: {e}", file=sys.stderr)
        print(
            "Please ensure you have installed the required packages (pip install vllm)",
            file=sys.stderr)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user.")
    except FileNotFoundError:
        print(
            f"\n❌ Error: '{sys.executable} -m vllm.entrypoints.openai.api_server' not found.",
            file=sys.stderr)
        print(
            "Please ensure you are in the correct environment and have vLLM installed.",
            file=sys.stderr)


def run_profiler(profile_args, additional_args):
    """Constructs and runs the nsys profiler command."""
    nsys_command = [
        "nsys", "profile", "-t", "cuda,nvtx,osrt", "--force-overwrite", "true", "-o",
        profile_args.output
    ]

    benchmark_command = [
        sys.executable,  # Use the project's python interpreter
        "benchmark/open_ai_api.py"
    ]

    # Add any additional arguments for the benchmark script
    if additional_args:
        if additional_args and additional_args[0] == '--':
            additional_args.pop(0)
        benchmark_command.extend(additional_args)

    command = nsys_command + benchmark_command

    print("\n🕵️  Running Nsight profiler with the following command:")
    print("  " + " ".join(command))

    try:
        subprocess.run(command, check=True)
        print(f"\n✅ Profiling complete. Report saved to '{profile_args.output}.nsys-rep'")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error during profiling: {e}", file=sys.stderr)
    except FileNotFoundError:
        print("\n❌ Error: 'nsys' command not found.", file=sys.stderr)
        print("Please ensure NVIDIA Nsight Systems is installed and in your system's PATH.", file=sys.stderr)


def main():
    """
    Main function to parse arguments and launch the vLLM server or profiler.
    """
    parser = argparse.ArgumentParser(
        description=
        "A plug-and-play script to launch the vLLM server or Nsight profiler."
    )
    
    # Server mode arguments
    server_group = parser.add_argument_group('Server Mode (Default)')
    server_group.add_argument(
        "--model",
        type=str,
        help=
        "Directly specify the model to serve, a new model will be created if one does not exist.")
    # Profiler mode arguments
    profile_group = parser.add_argument_group('Profiler Mode')
    profile_group.add_argument(
        "--profile",
        action='store_true',
        help="Enable profiler mode to run the benchmark with Nsight Systems."
    )
    profile_group.add_argument(
        "--profile-output",
        type=str,
        dest='output',
        default="benchmark/my_annotated_report",
        help="Output file name for the Nsight report (without extension)."
    )

    parser.add_argument(
        'additional_args',
        nargs=argparse.REMAINDER,
        help=
        "Additional arguments to pass to the server or the benchmark script (e.g., --num-prompts 50)."
    )

    args = parser.parse_args()

    if args.profile:
        run_profiler(args, args.additional_args)
    else:
        model_name = args.model
        if not model_name:
            display_menu()
            model_name = get_model_choice()

        if model_name:
            run_server(model_name, args.additional_args)


if __name__ == "__main__":
    main()
