import subprocess
import sys
import argparse
import os
from config_handler import load_config
from logger import logger

# A list of common models for user convenience
SUPPORTED_MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Llama-2-7b-chat-hf",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "google/gemma-2b-it",
]


def display_menu():
    """Prints the model selection menu."""
    logger.info("\n--- vLLM Server Launcher ---")
    logger.info("Please choose a model to serve:")
    for i, model in enumerate(SUPPORTED_MODELS):
        logger.info(f"  {i + 1}. {model}")
    logger.info(f"  {len(SUPPORTED_MODELS) + 1}. Enter a custom HuggingFace model name")
    logger.info("  0. Exit")


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
                        logger.error("Custom model name cannot be empty.")
                        continue
                return SUPPORTED_MODELS[choice - 1]
            else:
                logger.error(
                    f"Invalid choice. Please enter a number between 0 and {len(SUPPORTED_MODELS) + 1}."
                )
        except ValueError:
            logger.error("Invalid input. Please enter a number.")
        except IndexError:
            logger.error("Invalid choice. Please select a valid option from the menu.")


def build_command_from_config(config, additional_args):
    """Builds a command list from a configuration dictionary."""
    command = []
    for key, value in config.items():
        if value is not None:
            # Convert key to a command-line argument
            arg = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    command.append(arg)
            elif isinstance(value, list):
                for item in value:
                    command.extend([arg, str(item)])
            else:
                command.extend([arg, str(value)])

    # Override with additional args
    if additional_args:
        # The vllm entrypoint script doesn't like the '--' separator
        if additional_args and additional_args[0] == '--':
            additional_args.pop(0)

        # Simple override: if an arg is in additional_args, it replaces the config value
        # This is a simplification. A more robust solution would parse additional_args properly.
        for i, arg in enumerate(additional_args):
            if arg.startswith('--'):
                arg_name = arg[2:].replace('-', '_')
                # remove existing from command
                for j, cmd_arg in enumerate(command):
                    if cmd_arg == arg:
                        # remove argument and its value
                        command.pop(j)
                        command.pop(j)
                        break
        command.extend(additional_args)

    return command


def run_server(config, additional_args):
    """Constructs and runs the vLLM server command."""
    server_command_args = build_command_from_config(
        config.get("server_config", {}), additional_args
    )
    command = [
        sys.executable,  # Use the same python interpreter
        "-m",
        "vllm.entrypoints.openai.api_server",
    ] + server_command_args

    logger.info("\n🚀 Starting vLLM server with the following command:")
    # Format for readability
    logger.info("  " + " ".join(command))
    logger.info("\nTo stop the server, press Ctrl+C in this terminal.")

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"\n❌ Error starting the server: {e}")
        logger.error(
            "Please ensure you have installed the required packages (pip install vllm)")
    except KeyboardInterrupt:
        logger.warning("\n🛑 Server stopped by user.")
    except FileNotFoundError:
        logger.error(
            f"\n❌ Error: '{sys.executable} -m vllm.entrypoints.openai.api_server' not found.")
        logger.error(
            "Please ensure you are in the correct environment and have vLLM installed.")


def run_profiler(config, profile_args, additional_args):
    """Constructs and runs the nsys profiler command."""
    nsys_command = [
        "nsys", "profile", "-t", "cuda,nvtx,osrt", "--force-overwrite", "true", "-o",
        profile_args.output
    ]

    benchmark_command_args = build_command_from_config(
        config.get("benchmark_config", {}), additional_args
    )
    benchmark_command = [
        sys.executable,  # Use the project's python interpreter
        "open_ai_api.py"
    ] + benchmark_command_args

    command = nsys_command + benchmark_command

    logger.info("\n🕵️  Running Nsight profiler with the following command:")
    logger.info("  " + " ".join(command))

    try:
        subprocess.run(command, check=True)
        logger.info(f"\n✅ Profiling complete. Report saved to '{profile_args.output}.nsys-rep'")
    except subprocess.CalledProcessError as e:
        logger.error(f"\n❌ Error during profiling: {e}")
    except FileNotFoundError:
        logger.error("\n❌ Error: 'nsys' command not found.")
        logger.error("Please ensure NVIDIA Nsight Systems is installed and in your system's PATH.")


def main():
    """
    Main function to parse arguments and and launch the vLLM server or profiler.
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
        "Directly specify the model to serve, overriding the config file.")
    
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

    # Load configuration
    config = load_config()
    if not config:
        sys.exit(1)

    if args.profile:
        run_profiler(config, args, args.additional_args)
    else:
        # Handle model override
        model_name = args.model
        if not model_name:
            # only show menu if model is not passed via CLI
            display_menu()
            model_name = get_model_choice()
            if model_name is None:
                sys.exit(0) # User chose to exit

        if model_name:
            config['server_config']['model'] = model_name

        run_server(config, args.additional_args)


if __name__ == "__main__":
    main()
