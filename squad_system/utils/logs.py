from logging import config as logging_config

from yaml import FullLoader, load


def initialize_logging(config_path: str):
    with open(config_path) as yaml_fh:
        config_description = load(yaml_fh, Loader=FullLoader)
        logging_config.dictConfig(config_description)
