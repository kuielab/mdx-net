import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf

dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="evaluation.yaml")
def main(config: DictConfig):
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from src.evaluation.eval import evaluation
    from src.utils import utils

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    evaluation(config)


if __name__ == "__main__":
    main()
