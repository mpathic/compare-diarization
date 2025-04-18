import argparse
import logging
import os
import json
import sys
import time

from dotenv import load_dotenv # Import load_dotenv
load_dotenv() # Load environment variables from .env file

from src.s3_fh import download_file
from src.processors import td_revai, td_assemblyai, td_opensource
# from src.comparison.metrics import calculate_all_metrics

## USAGE (from root directory of project):
## python3 -m src.main


logging.basicConfig(
	# level=logging.INFO,
	level=logging.DEBUG,
	format='%(asctime)s %(name)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def process_audio(audio_file):
	logger.info("Starting processors ...")

	results = {
		'revai' : td_revai.process(audio_file),			# do revai
		'assemblyai' : td_assemblyai.process(audio_file), # do aai
		'opensource' : td_opensource.process(audio_file) # do open source
	}
	logger.info("Finished processing audio file through sep methods.")
	return results
	

def main():

	logger.info("starting ...")

	# run against a single audio file, use something already on disk
	audio_file = "g2v2sfwfQ84_Brief_intervention_Dave.mp3"

	results = process_audio(audio_file)


	logger.info("Workflow finished.")


if __name__ == "__main__":
	main()