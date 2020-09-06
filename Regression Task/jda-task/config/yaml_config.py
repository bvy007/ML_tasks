
"""  Code to load the yaml files  """

import yaml
import os

from config import global_config as glob

arguments_path = glob.config_dir
verbose = False

with open(os.path.join(arguments_path, 'arguments.yaml'), 'r') as stream:
    try:
        model_parameters = yaml.load(stream, Loader=yaml.FullLoader)
        if verbose:
            print("Read arguments.yaml:" + str(model_parameters))
    except yaml.YAMMLError as exc:
        print(exc)