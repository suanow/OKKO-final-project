import logging
from dynaconf import Dynaconf

# specify logging level
logging.basicConfig(level=logging.INFO)


settings = Dynaconf(settings_file='conf/settings.toml')