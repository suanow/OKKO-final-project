import logging
from lightfm.data import Dataset
from dynaconf import Dynaconf

# specify logging level
logging.basicConfig(level=logging.INFO)


settings = Dynaconf(settings_file='conf/settings.toml')

dataset = Dataset()