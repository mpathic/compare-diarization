import argparse
import logging
import os
import json
import sys
import time
import csv

from dotenv import load_dotenv
load_dotenv() # load ennvars from .env file

# from src.s3_fh import download_file
from src.processors import td_revai, td_assemblyai, td_opensource
from src.download_videos import download


# from src.comparison.metrics import calculate_all_metrics
import Levenshtein


## USAGE (from root directory of project):
## python3 -m src.main


logging.basicConfig(
	level=logging.INFO,
	# level=logging.DEBUG,
	format='%(asctime)s %(name)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def process_audio(audio_file):
	logger.info("Starting Audio Processors ...")

	results = {
		'revai' : td_revai.process(audio_file),			# do revai
		'assemblyai' : td_assemblyai.process(audio_file), # do aai
		'opensource' : td_opensource.process(audio_file) # do open source
	}
	logger.info("Finished processing audio file through sep methods.")

	return results


def examine(results):
	logger.info("Evaluation Process ...")

	# first get the WER 

	'''
	I define a new error rate, called the WhoSaidWhat Error Rate (WSWER)
	exact chronology doesnt matter for our use case, 
	what really only matters is who said what in the conversation
	for a lot of our cases right now, exact ordering of what was said 
	is not too important either.

	Exact timestamps of when things were said is not important for my domain
	and use case. What's more important for my use case is the overall flow
	of the conversation, specifically, who said what.

	This newly defined error rate measures discrepancies in content that each
	speaker used in the conversation.
	'''


def load_ground_truth(gt_filepath):
	# store the transcript data by transcript_id
	logger.info("Reading GT metadata and importing ground truth data ...")
	transcripts = {}

	with open(gt_filepath, 'r') as file:
		csv_reader = csv.DictReader(file)
		
		for row in csv_reader:

			transcript_id = row['transcript_id']
			video_title = row['video_title']
			video_url = row['video_url']
			mi_quality = row['mi_quality'] # from 23 low and 100 high
			utterance_id = int(row['utterance_id'])
			speaker = row['interlocutor']
			text = row['utterance_text'].strip()

			if transcript_id not in transcripts.keys():
				logger.debug(f"Adding gt video {video_url}:{video_title}...")

				transcripts[transcript_id] = {
					'transcript_id' : transcript_id,
					'video_title' : video_title,
					'video_url' : video_url,
					'mi_quality' : mi_quality,
					'all_utterances' : [],
					'who_said_what' : {}
				}

			transcripts[transcript_id]['all_utterances'].append(text) # list of texts

			if speaker not in transcripts[transcript_id]['who_said_what'].keys():
				transcripts[transcript_id]['who_said_what'][speaker] = []

			transcripts[transcript_id]['who_said_what'][speaker].append(text)

	# now go through and consolidate them to the correct output
	for transcript_id, item in transcripts.items():

		text_paragraph = ' '.join(item['all_utterances'])
		transcripts[transcript_id]['transcript'] = text_paragraph # continuous transript

		for speaker, their_texts in item['who_said_what'].items():
			transcripts[transcript_id]['who_said_what'][speaker] = ' '.join(their_texts) # overwrite

	logger.info("Done importing ground truth data.")
	return transcripts



def main():
	logger.info("starting ...")
	comparison_results = {} # collection of all the results

	# load the ground truth transcripts
	ground_truth_filepath = 'evaluation_data/AnnoMI-full-export-ground-truth.csv'
	ground_truth = load_ground_truth(ground_truth_filepath) # this returns a dictionary


	for transcript_id in list(ground_truth.keys()):
		evaluation = {} # make a new one

		info = ground_truth[transcript_id]
		logger.info(f"{info['video_title']}:")
		evaluation['video_title'] = info['video_title']

		gt_transcript = ground_truth[transcript_id]['transcript'] # continuous transcript
		gt_wsw_transcript = ground_truth[transcript_id]['who_said_what'] # whosaidwhat transcript

		logger.info(f"\tGT transcript: {gt_transcript}")
		gt_speakers = ground_truth[transcript_id]['who_said_what'].keys()

		evaluation['gt_speakers'] = gt_speakers
		evaluation['gt_num_speakers'] = len(gt_speakers)
		

		#
		# get the file info, pull the file and process it.
		#
		video_url = info['video_url']
		downloaded_file = download(video_url)
		ground_truth[transcript_id]['downloaded_filepath'] = downloaded_file # keep track of its path



		#
		# get diarized transript for each of 3 methods
		#
		results = process_audio(downloaded_file)



		# EVAL
		# now, compare each with the ground truth item
		# comparison_results[transcript_id]['eval'] = {} # a set for eval
		# evaluation[]
		processor_methods = results.keys()


		for processor in processor_methods:

			logger.info(f"Evaluating GT against processor: {processor}")
			evaluation[processor] = {}

			#
			# WER
			#
			# get levenstein distance for the continuous transcript
			processor_transcript = results[processor]['continuous_transcript']
			logger.info(f"\t{processor} transcript: {processor_transcript}")

			# Calculate the distance
			distance = Levenshtein.distance(gt_transcript, processor_transcript)
			logger.info(f"distance: {distance}")  # Output: 3
			evaluation[processor]['transcript_distance'] = distance

			#
			# WSWER ~ DER
			#
			# get levenstein distance for the diarized groups
			processor_wsw_transcript = results[processor]['whosaidwhat_transcript'] # wsw transcript
			processor_speakers = results[processor]['whosaidwhat_transcript'].keys()

			logger.info(f"GT contains {len(gt_speakers)} speakers.")
			logger.info(f"Processor {processor} detected {len(processor_speakers)} speakers.")
			evaluation[processor]['speakers'] = processor_speakers
			evaluation[processor]['num_speakers'] = len(processor_speakers)


			# log the outputs to aws, create an augmented GT file with the narrator's intro
			# compare the number of speakers.
			logger.info("GT transript items:")
			for k,v in gt_wsw_transcript.items():
				print(f"{k}: {v}\n")


			logger.info("WSW transript items:")
			for k,v in processor_wsw_transcript.items():
				print(f"{k}: {v}\n")




		comparison_results[transcript_id] = evaluation


	logger.info("Workflow finished.")
	for k,v in comparison_results.items():
		print(k,v)



if __name__ == "__main__":
	main()

