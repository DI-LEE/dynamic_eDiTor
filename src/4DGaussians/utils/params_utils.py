import sys


def merge_hparams(args, config):
    """
    Merge values from a config file into argparse Namespace, but DO NOT override
    keys that were explicitly provided via the CLI (detected from sys.argv).
    """
    # Collect CLI-provided long option names (e.g., "layer_range" for "--layer_range")
    cli_keys = set()
    for token in sys.argv[1:]:
        if token.startswith("--"):
            name = token.lstrip("-").split("=", 1)[0]
            cli_keys.add(name)

    params = ["OptimizationParams", "ModelHiddenParams", "ModelParams", "PipelineParams"]
    for param in params:
        if param in config.keys():
            for key, value in config[param].items():
                # Only set from config if CLI did not explicitly specify this key
                if hasattr(args, key) and key not in cli_keys:
                    setattr(args, key, value)

    return args