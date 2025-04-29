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
		'revai' : td_revai.process(audio_file),           # do revai
		'assemblyai' : td_assemblyai.process(audio_file), # do aai
		'opensource' : td_opensource.process(audio_file)  # do open source
	}
	logger.info("Finished processing audio file through sep methods.")

	return results


def calculate_wer(reference, hypothesis):
	"""
	Calculate Word Error Rate (WER): Levenshtein distance normalized by reference length.
	
	Args:
		reference (str): The ground truth text
		hypothesis (str): The predicted text from processor
		
	Returns:
		float: WER value (between 0 and potentially > 1.0)
	"""
	# Calculate Levenshtein distance
	distance = Levenshtein.distance(reference, hypothesis)
	
	# Count words in reference
	ref_word_count = len(reference.split())
	
	# Avoid division by zero
	if ref_word_count == 0:
		return float('inf')
	
	# Calculate WER
	wer = distance / ref_word_count
	
	return wer


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


def find_best_speaker_matches(gt_wsw_transcript, processor_wsw_transcript):
	"""
	Find the best matching pairs between ground truth speakers and processor speakers
	based on text similarity (Levenshtein distance)
	
	Args:
		gt_wsw_transcript: Dictionary of {speaker: text} from ground truth
		processor_wsw_transcript: Dictionary of {speaker: text} from processor
		
	Returns:
		list of tuples: [(gt_speaker, processor_speaker), ...]
	"""
	matches = []
	distance_matrix = {}
	
	# Calculate distance between each pair of speakers
	for gt_speaker, gt_text in gt_wsw_transcript.items():
		distance_matrix[gt_speaker] = {}
		for proc_speaker, proc_text in processor_wsw_transcript.items():
			distance = Levenshtein.distance(gt_text, proc_text)
			distance_matrix[gt_speaker][proc_speaker] = distance
	
	# Assign each ground truth speaker to best matching processor speaker
	gt_speakers = list(gt_wsw_transcript.keys())
	proc_speakers = list(processor_wsw_transcript.keys())
	
	# Handle case when different number of speakers
	if len(gt_speakers) != len(proc_speakers):
		logger.warning(f"Different number of speakers: GT has {len(gt_speakers)}, processor has {len(proc_speakers)}")
		
		# If processor detected fewer speakers, we can only match those
		if len(proc_speakers) < len(gt_speakers):
			# For each processor speaker, find best matching GT speaker
			assigned_gt = set()
			for proc_speaker in proc_speakers:
				best_gt = None
				best_distance = float('inf')
				
				for gt_speaker in gt_speakers:
					if gt_speaker not in assigned_gt:
						distance = min([distance_matrix[gt_speaker][proc_speaker] 
									  for gt_speaker in gt_speakers 
									  if gt_speaker not in assigned_gt])
						
						if distance < best_distance:
							best_distance = distance
							best_gt = gt_speaker
				
				matches.append((best_gt, proc_speaker))
				assigned_gt.add(best_gt)
		else:
			# If processor detected more speakers, match each GT speaker to best processor speaker
			assigned_proc = set()
			for gt_speaker in gt_speakers:
				best_proc = None
				best_distance = float('inf')
				
				for proc_speaker in proc_speakers:
					if proc_speaker not in assigned_proc:
						distance = distance_matrix[gt_speaker][proc_speaker]
						
						if distance < best_distance:
							best_distance = distance
							best_proc = proc_speaker
				
				matches.append((gt_speaker, best_proc))
				assigned_proc.add(best_proc)
	else:
		# When equal number of speakers, use Hungarian algorithm for optimal assignment
		try:
			from scipy.optimize import linear_sum_assignment
			import numpy as np
			
			# Create cost matrix
			cost_matrix = []
			for gt_speaker in gt_speakers:
				row = [distance_matrix[gt_speaker][proc_speaker] for proc_speaker in proc_speakers]
				cost_matrix.append(row)
			
			# Apply Hungarian algorithm
			row_ind, col_ind = linear_sum_assignment(np.array(cost_matrix))
			
			# Create matches based on the assignment
			for i, j in zip(row_ind, col_ind):
				matches.append((gt_speakers[i], proc_speakers[j]))
		except ImportError:
			# Fallback if scipy not available: greedy assignment
			assigned_proc = set()
			for gt_speaker in gt_speakers:
				best_proc = None
				best_distance = float('inf')
				
				for proc_speaker in proc_speakers:
					if proc_speaker not in assigned_proc:
						distance = distance_matrix[gt_speaker][proc_speaker]
						
						if distance < best_distance:
							best_distance = distance
							best_proc = proc_speaker
				
				matches.append((gt_speaker, best_proc))
				assigned_proc.add(best_proc)
	
	return matches, distance_matrix


def main():
	logger.info("starting ...")
	comparison_results = {} # collection of all the results

	# load the ground truth transcripts
	ground_truth_filepath = 'evaluation_data/AnnoMI-full-export-ground-truth.csv'
	ground_truth = load_ground_truth(ground_truth_filepath)

	# for simplicity and compute sake, only sample a few of the transcripts
	transcripts_to_sample = list(ground_truth.keys())

	random.seed(42)
	num_to_sample = int(len(transcripts_to_sample) * 0.33)
	sampled_transcripts = random.sample(transcripts_to_sample, num_to_sample)
	logger.info(f"Transcripts to be sampled: {sampled_transcripts}")

	missing_audio = [89]
	has_narrator = [0, 24, 34, 44, 63, 67, 72, 99, 106, 109, 112, 123, 129]
	accent_is_british = [16, 64, 65, 84, 117, 118, 131]
	accent_is_southern_us = [3, 57, 59, 103]

	# Remove both narrator and missing IDs from the list of transcripts
	filtered_transcripts = list(set(sampled_transcripts) - set(has_narrator) - set(missing_audio))
	logger.info(f"Filtered transcripts (no narrator, no missing audio): {filtered_transcripts}")
	
	# Log the British and Southern accent IDs for reference
	british_in_filtered = list(set(filtered_transcripts).intersection(set(accent_is_british)))
	southern_in_filtered = list(set(filtered_transcripts).intersection(set(accent_is_southern_us)))
	logger.info(f"British accent IDs in filtered set: {british_in_filtered}")
	logger.info(f"Southern accent IDs in filtered set: {southern_in_filtered}")


	# download the audio files locally in the project
	audio_filepaths = s3_download_files(filtered_transcripts) # {transcript_id : filepath}

	logger.debug("Pulled the following files from S3:")
	for k,v in audio_filepaths.items():
		logger.debug(f"{k} : {v}")


	# collect the transcript ids with narrators etc:
	transcripts_diffnum_speakers = []

	# iterate over each transcript_id
	i=1
	for transcript_id in filtered_transcripts:
		logger.info(f"Starting to process transcript {transcript_id} ...")
		logger.info(f"\t item {i} / {len(filtered_transcripts)}:")
		i+=1

		# parse audio info
		audio_info = audio_filepaths[transcript_id]
		audio_filepath = audio_info['filepath']
		audio_duration = audio_info['duration']

		# parse ground truth info
		transcript_info = ground_truth[transcript_id]
		title = transcript_info['video_title']
		logger.info(f"Title: {title}:")

		gt_speakers = list(transcript_info['who_said_what'].keys())
		gt_transcript = transcript_info['transcript'] # continuous transcript
		gt_wsw_transcript = transcript_info['who_said_what'] # whosaidwhat transcript
		
		evaluation = {} # make a new one
		evaluation['audio_duration'] = audio_duration
		evaluation['gt_speakers'] = gt_speakers
		evaluation['gt_num_speakers'] = len(gt_speakers)
		evaluation['downloaded_filepath'] = audio_filepath # keep track of its path

		evaluation['has_narrator'] = False
		if int(transcript_id) in has_narrator:
			evaluation['has_narrator'] = True

		evaluation['accent'] = "American"
		if int(transcript_id) in accent_is_british:
			evaluation['accent'] = "British"
		elif int(transcript_id) in accent_is_southern_us:
			evaluation['accent'] = "Southern US"


		# Get word count of ground truth transcript for WER calculation
		gt_word_count = len(gt_transcript.split())
		evaluation['gt_word_count'] = gt_word_count

		# Get word counts for each speaker in ground truth for WSW-WER calculation
		gt_speaker_word_counts = {}
		for speaker, text in gt_wsw_transcript.items():
			gt_speaker_word_counts[speaker] = len(text.split())
		evaluation['gt_speaker_word_counts'] = gt_speaker_word_counts


		#
		# PROCESS: get diarized transcript for each of 3 methods
		#
		results = process_audio(audio_filepath)


		# EVALUATION
		# now, compare each with the ground truth item
		processor_methods = results.keys()
		for processor in processor_methods:

			logger.info(f"Evaluating GT against processor: {processor}")
			evaluation[processor] = {}

			#
			# TRANSCRIPT EVALUATION - Raw distance and WER
			#
			processor_transcript = results[processor]['continuous_transcript']

			# Calculate the raw Levenshtein distance
			distance = Levenshtein.distance(gt_transcript, processor_transcript)
			logger.info(f"Continuous transcript distance: {distance}")
			evaluation[processor]['transcript_distance'] = distance
			
			# Calculate WER (normalized by word count)
			transcript_wer = calculate_wer(gt_transcript, processor_transcript)
			logger.info(f"Continuous transcript WER: {transcript_wer:.4f}")
			evaluation[processor]['transcript_wer'] = transcript_wer

			#
			# SPEAKER DIARIZATION EVALUATION
			#
			processor_wsw_transcript = results[processor]['whosaidwhat_transcript'] # wsw transcript
			processor_speakers = list(results[processor]['whosaidwhat_transcript'].keys())

			logger.info(f"GT contains {len(gt_speakers)} speakers.")
			logger.info(f"Processor {processor} detected {len(processor_speakers)} speakers.")
			evaluation[processor]['speakers'] = processor_speakers
			evaluation[processor]['num_speakers'] = len(processor_speakers)

			# Find optimal speaker matches
			speaker_matches, distance_matrix = find_best_speaker_matches(
				gt_wsw_transcript, processor_wsw_transcript
			)

			# Now evaluate using the matched speakers
			evaluation[processor]['wsw_distance'] = {}
			evaluation[processor]['wsw_wer'] = {}
			evaluation[processor]['speaker_matches'] = []
			
			for i, (gt_speaker, p_speaker) in enumerate(speaker_matches):
				logger.info(f"Matched: GT Speaker {gt_speaker} with {processor} Speaker {p_speaker}")
				
				gt_wsw_segment = gt_wsw_transcript[gt_speaker]
				p_wsw_segment = processor_wsw_transcript[p_speaker]
				
				# Get word count for this speaker's ground truth
				speaker_word_count = gt_speaker_word_counts.get(gt_speaker, 0)
				
				# Calculate raw Levenshtein distance
				wsw_distance = Levenshtein.distance(gt_wsw_segment, p_wsw_segment)
				logger.info(f"WSW distance: {wsw_distance}")
				
				# Calculate WER for this speaker
				if speaker_word_count > 0:
					wsw_wer = wsw_distance / speaker_word_count
					logger.info(f"WSW WER: {wsw_wer:.4f}")
				else:
					wsw_wer = float('inf')
					logger.warning(f"Speaker {gt_speaker} has no words in ground truth, cannot calculate WER")
				
				evaluation[processor]['wsw_distance'][i] = {
					'speaker_pair': {
						'gt_speaker': gt_speaker,
						'p_speaker': p_speaker
					},
					'distance': wsw_distance,
					'word_count': speaker_word_count,
					'wer': wsw_wer
				}
				
				evaluation[processor]['wsw_wer'][i] = wsw_wer
				
				evaluation[processor]['speaker_matches'].append({
					'gt_speaker': gt_speaker,
					'p_speaker': p_speaker,
					'distance': wsw_distance,
					'wer': wsw_wer
				})
			
			# Calculate average WSW-WER across all speakers
			wsw_wer_values = [data['wer'] for speaker_idx, data in evaluation[processor]['wsw_distance'].items() 
							 if data['wer'] != float('inf')]
			
			if wsw_wer_values:
				avg_wsw_wer = sum(wsw_wer_values) / len(wsw_wer_values)
				evaluation[processor]['average_wsw_wer'] = avg_wsw_wer
				logger.info(f"Average WSW-WER across all speakers: {avg_wsw_wer:.4f}")
			else:
				evaluation[processor]['average_wsw_wer'] = None
				logger.warning("Could not calculate average WSW-WER (no valid values)")
			
			# Also store the complete distance matrix for reference
			evaluation[processor]['distance_matrix'] = distance_matrix


		# push the eval object for all three processors to the result set
		comparison_results[transcript_id] = evaluation
		logger.info(f"Evaluation for transcript {transcript_id}:")
		logger.info(json.dumps(evaluation, indent=2))


	logger.info("Workflow finished.")
	# Don't print the full results to avoid log overflow
	# logger.info(json.dumps(comparison_results, indent=2))

	write_results_to_file(comparison_results, 'out/evaluation_results_with_wer.json')

	logger.info("Files with diff number of speakers:")
	for item in transcripts_diffnum_speakers:
		logger.info(item)


if __name__ == "__main__":
	main()