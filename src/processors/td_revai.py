import os
import json
import logging

from rev_ai import apiclient
from time import sleep

logger = logging.getLogger(__name__) # respect mains loglevel

REVAI_TOKEN = os.getenv('REVAI_TOKEN', None)


if __name__ == '__main__':

	logger.info("transcribing via REV AI ...")


