import argparse
import logging
import os
import json
import sys
import time
import csv
import random

from dotenv import load_dotenv
load_dotenv() # load ennvars from .env file

from src.s3_fh import s3_download_files
from src.processors import td_revai, td_assemblyai, td_opensource
from src.download_videos import download


# from src.comparison.metrics import calculate_all_metrics
import Levenshtein
from difflib import unified_diff


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
	logger.info("Reading GT metadata and importing ground truth data (deduplicating utterances)...")
	transcripts = {}
	processed_utterances = set() # Keep track of (transcript_id, utterance_id) pairs

	try:
		with open(gt_filepath, 'r', encoding='utf-8') as file:
			csv_reader = csv.DictReader(file)

			for i, row in enumerate(csv_reader):
				try:
					transcript_id = row.get('transcript_id')
					utterance_id_str = row.get('utterance_id')

					# Basic validation
					if not transcript_id or utterance_id_str is None:
						logger.warning(f"Row {i+1}: Missing transcript_id or utterance_id. Skipping.")
						continue

					try:
						utterance_id = int(utterance_id_str)
					except ValueError:
						logger.warning(f"Row {i+1}: Invalid utterance_id '{utterance_id_str}'. Skipping.")
						continue

					# --- Deduplication Check ---
					utterance_key = (transcript_id, utterance_id)
					if utterance_key in processed_utterances:
						logger.debug(f"Row {i+1}: Duplicate utterance {utterance_key}. Skipping text append.")
						continue

					processed_utterances.add(utterance_key)
					# --------------------------

					video_title = row.get('video_title', 'N/A')
					video_url = row.get('video_url', 'N/A')
					mi_quality = row.get('mi_quality', 'N/A')
					speaker = row.get('interlocutor', 'Unknown')
					text = row.get('utterance_text', '').strip()

					# Initialize transcript entry if it's the first time seeing this transcript_id
					if transcript_id not in transcripts:
						logger.debug(f"Adding gt video {video_url}:{video_title}...")
						transcripts[transcript_id] = {
							'transcript_id' : transcript_id,
							'video_title' : video_title,
							'video_url' : video_url,
							'mi_quality' : mi_quality,
							'all_utterances' : [],
							'who_said_what' : {}
						}

					# Append utterance text (only happens if not a duplicate key)
					transcripts[transcript_id]['all_utterances'].append(text)

					# Append speaker-specific text
					if speaker not in transcripts[transcript_id]['who_said_what']:
						transcripts[transcript_id]['who_said_what'][speaker] = []
					transcripts[transcript_id]['who_said_what'][speaker].append(text)

				except KeyError as e:
					 logger.error(f"Row {i+1}: Missing expected column key: {e}. Check CSV header.")
				except Exception as e:
					 logger.error(f"Row {i+1}: Unexpected error processing row: {e}")

	except FileNotFoundError:
		logger.error(f"Error: File not found at {gt_filepath}")
		return {}
	except Exception as e:
		logger.error(f"Error opening or reading file {gt_filepath}: {e}")
		return {}

	# Consolidate texts
	logger.info("Consolidating transcript texts...")
	for transcript_id, item in transcripts.items():
		# Sort utterances? If the CSV isn't guaranteed to be sorted by utterance_id,
		# might want to store (utterance_id, text) tuples and sort before joining.
		# For now, assuming order is preserved or OK as is.
		text_paragraph = ' '.join(item['all_utterances'])
		transcripts[transcript_id]['transcript'] = text_paragraph # continuous transcript

		for speaker, their_texts in item['who_said_what'].items():
			transcripts[transcript_id]['who_said_what'][speaker] = ' '.join(their_texts) # overwrite with consolidated text

	logger.info("Done importing and deduplicating ground truth data.")
	return transcripts


def write_results_to_file(results, output_filepath='out/evaluation_results.json'):
	"""
	Writes comparison results to a JSON file on disk.
	"""
	logger.info(f"Writing results to {output_filepath}...")
	try:
		with open(output_filepath, 'w', encoding='utf-8') as f:
			json.dump(results, f, indent=4)
		logger.info(f"Results successfully written to {output_filepath}")
		return output_filepath
	except Exception as e:
		logger.error(f"Error writing results to file: {e}")
		return None


def main():
	logger.info("starting ...")
	comparison_results = {} # collection of all the results

	# load the ground truth transcripts
	ground_truth_filepath = 'evaluation_data/AnnoMI-full-export-ground-truth.csv'
	ground_truth = load_ground_truth(ground_truth_filepath)

	# for simplicity and compute sake, only sample a few of the transcripts
	# for now
	transcripts_to_sample = list(ground_truth.keys())

	random.seed(0)
	num_to_sample = int(len(transcripts_to_sample) * 0.15)
	sampled_transcripts = random.sample(transcripts_to_sample, num_to_sample)
	logger.info(f"Transcripts to be sampled: {sampled_transcripts}")

	has_narrator = [34, 67, 132, 7, 123, 129]
	filtered_transcripts = list(set(sampled_transcripts) - set(has_narrator))
	sampled_transcripts = filtered_transcripts # rename for laziness
	logger.info(f"Filtered transcripts (no narrator): {sampled_transcripts}")


	# download the audio files locally in the project
	# audio_filepaths = s3_download_files() # {transcript_id : filepath}
	audio_filepaths = s3_download_files(sampled_transcripts) # {transcript_id : filepath}

	logger.debug("Pulled the following files from S3:")
	for k,v in audio_filepaths.items():
		logger.debug(f"{k} : {v}")


	# collect the transcript ids with narrators etc:
	transcripts_diffnum_speakers = []

	# iterate over each transcript_id
	i=1
	for transcript_id in sampled_transcripts:
		logger.info(f"Starting to process transcript {transcript_id} ...")
		logger.info(f"\t item {i} / {num_to_sample}:")
		i+=1

		# parse info
		audio_info = audio_filepaths[transcript_id]
		audio_filepath = audio_info['filepath']
		audio_duration = audio_info['duration']
		title = audio_info['video_title']

		logger.info(f"Title: {title}:")

		evaluation = {} # make a new one
		evaluation['audio_duration'] = audio_duration

		# stash the ground truth transcripts
		gt_transcript = ground_truth[transcript_id]['transcript'] # continuous transcript
		gt_wsw_transcript = ground_truth[transcript_id]['who_said_what'] # whosaidwhat transcript
		logger.info(f"\tGT transcript: {gt_transcript}")

		# stash the number of ground truth speakers
		gt_speakers = list(ground_truth[transcript_id]['who_said_what'].keys())
		evaluation['gt_speakers'] = gt_speakers
		evaluation['gt_num_speakers'] = len(gt_speakers)
		

		#
		# reference the audio file for this transcript
		#
		audio_filepath = audio_filepaths[transcript_id]
		evaluation['downloaded_filepath'] = audio_filepath # keep track of its path


		#
		# get diarized transript for each of 3 methods
		#
		results = process_audio(audio_filepath)


		# EVALUATION
		# now, compare each with the ground truth item
		processor_methods = results.keys()
		for processor in processor_methods:

			logger.info(f"Evaluating GT against processor: {processor}")
			evaluation[processor] = {}

			#
			# WER
			#
			# get levenstein distance for the continuous transcript
			processor_transcript = results[processor]['continuous_transcript']
			# logger.info(f"\t{processor} transcript: {processor_transcript}")

			# Calculate the distance
			distance = Levenshtein.distance(gt_transcript, processor_transcript)
			logger.info(f"continuous transcript distance: {distance}")  # Output: 3
			evaluation[processor]['transcript_distance'] = distance

			#
			# WSWER ~DER
			#
			# get levenstein distance for the diarized groups
			logger.info("\nprocessors results:")
			logger.info(results[processor])

			processor_wsw_transcript = results[processor]['whosaidwhat_transcript'] # wsw transcript
			processor_speakers = list(results[processor]['whosaidwhat_transcript'].keys())

			logger.info(f"GT contains {len(gt_speakers)} speakers.")
			logger.info(f"Processor {processor} detected {len(processor_speakers)} speakers.")
			evaluation[processor]['speakers'] = processor_speakers
			evaluation[processor]['num_speakers'] = len(processor_speakers)

			if len(gt_speakers) != len(processor_speakers):
				logger.warning("Different number of speakers detected (!) Skipping entire item from eval...")
				transcripts_diffnum_speakers.append({
					transcript_id: evaluation['video_title'],
					'processor' : processor,
					'gt_speakers' : gt_speakers,
					'processor_speakers' : processor_speakers
					})

				# sometimes processor might not get 2 speakers, just 1, so how to evaluate that?
				continue

			else:
				# processor_speakers = ['A', 'B']
				# gt_speakers = ['therapist', 'client']
				evaluation[processor]['wsw_distance'] = {}

				for i in range(len(gt_speakers)):
					logger.debug(f"gt_speaker index {i}")

					gt_speaker = gt_speakers[i] # gt sspeaker
					p_speaker = processor_speakers[i] # process
					logger.info(f"GT Speaker {gt_speaker} and {processor} Speaker {p_speaker}")

					gt_wsw_segment = gt_wsw_transcript[gt_speaker]
					p_wsw_segment = processor_wsw_transcript[p_speaker]

					wsw_distance = Levenshtein.distance(gt_wsw_segment, p_wsw_segment)
					logger.info(f"WSW distance: {wsw_distance}")  # Output: 3

					# Show word-level diff for debugging
					diff = '\n'.join(unified_diff(
						gt_wsw_segment.split(), 
						p_wsw_segment.split(), 
						fromfile='ground_truth', 
						tofile='processor_output', 
						lineterm=''
					))
					logger.debug(f"Text diff between GT and {processor}:\n{diff}")

					evaluation[processor]['wsw_distance'][i] = {
						'speaker_pair' : {
											'gt_speaker': gt_speaker, 
											'p_speaker': p_speaker},
						'distance' : wsw_distance
					}


		comparison_results[transcript_id] = evaluation
		logger.info(json.dumps(evaluation, indent=2))


	logger.info("Workflow finished.")
	logger.info(json.dumps(comparison_results, indent=2))

	write_results_to_file(comparison_results)

	logger.info("Files with diff number of speakers:")
	for item in transcripts_diffnum_speakers:
		logger.info(item)



if __name__ == "__main__":
	main()

