import os
import json
import logging 

import assemblyai as aai

logger = logging.getLogger(__name__) # respect mains loglevel


aai.settings.api_key  = os.getenv('AAI_KEY', None)


if __name__ == '__main__':

	logger.info("Assembly AI process ...")





