import logging
import sys

logger = logging.getLogger("RPG-DiffusionMaster")
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)


def change_debug(debug):
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
