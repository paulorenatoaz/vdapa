import yaml
from pathlib import Path


# Define os caminhos para os arquivos de configuração
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = BASE_DIR / 'config'
CONFIG_PATH = CONFIG_DIR / 'config.yaml'
PRIVATE_CONFIG_PATH = CONFIG_DIR / 'config_private.yaml'


def deep_update(base: dict, updates: dict):
    """
    Atualiza recursivamente o dicionário base com os valores do dicionário updates.
    """
    for key, value in updates.items():
        if isinstance(value, dict) and key in base:
            deep_update(base[key], value)
        else:
            base[key] = value

def load_config():
    """
    Carrega as configurações dos arquivos config.yaml e config_private.yaml (se existir),
    fazendo merge dos valores do arquivo privado sobre o público.
    """
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if PRIVATE_CONFIG_PATH.exists():
        with open(PRIVATE_CONFIG_PATH, 'r', encoding='utf-8') as f:
            private_config = yaml.safe_load(f)
        deep_update(config, private_config)

    return config

config = load_config()
