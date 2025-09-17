import argparse
import logging
import os
import json
import sys
import time
import csv
import random
import numpy as np
import torch
import librosa
import soundfile as sf # <-- Added for saving trimmed audio
from pydub import AudioSegment # <-- Added for robust audio handling

from dotenv import load_dotenv
load_dotenv() # load ennvars from .env file

from src.s3_fh import s3_download_files
from src.processors import td_revai, td_assemblyai, td_opensource
from src.download_videos import download
from src.accent_detector import AccentDetector

import Levenshtein
from difflib import unified_diff
import editdistance


## USAGE (from root directory of project):
## python3 -m src.main


logging.basicConfig(
	level=logging.INFO,
	# level=logging.DEBUG,
	format='%(asctime)s %(name)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def get_device():
	"""
	Detect and return the best available device (CUDA GPU if available, else CPU).
	"""
	if torch.cuda.is_available():
		device = torch.device('cuda')
		gpu_name = torch.cuda.get_device_name(0)
		gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
		logger.info(f"CUDA is available! Using GPU: {gpu_name}")
		logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
		logger.info(f"CUDA Version: {torch.version.cuda}")
		return device
	else:
		logger.info("CUDA is not available. Using CPU.")
		return torch.device('cpu')


def process_audio(audio_file):
	logger.info("Starting Audio Processors ...")

	results = {
		'revai' : td_revai.process(audio_file),          # do revai
		'assemblyai' : td_assemblyai.process(audio_file), # do aai
		'opensource' : td_opensource.process(audio_file)  # do open source
	}
	logger.info("Finished processing audio file through sep methods.")

	return results


def calculate_wer(reference, hypothesis):
	"""
	Calculate standard Word Error Rate (WER) using word-level Levenshtein distance.

	WER = (S + D + I) / N
	where:
	- S = substitutions
	- D = deletions
	- I = insertions
	- N = number of words in reference

	Args:
		reference (str): The ground truth text
		hypothesis (str): The predicted text from processor

	Returns:
		dict: Contains WER and detailed error breakdown
	"""
	# Tokenize into words and normalize
	ref_words = reference.lower().strip().split()
	hyp_words = hypothesis.lower().strip().split()

	# Handle empty cases
	if len(ref_words) == 0:
		if len(hyp_words) == 0:
			return {
				'wer': 0.0, 'substitutions': 0, 'insertions': 0,
				'deletions': 0, 'ref_length': 0, 'hyp_length': 0,
				'total_errors': 0
			}
		else:
			return {
				'wer': float('inf'), 'substitutions': 0, 'insertions': len(hyp_words),
				'deletions': 0, 'ref_length': 0, 'hyp_length': len(hyp_words),
				'total_errors': len(hyp_words)
			}

	# Initialize the distance matrix for dynamic programming
	d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.int32)

	# Initialize first row and column (deletion and insertion costs)
	for i in range(len(ref_words) + 1):
		d[i, 0] = i  # deletions
	for j in range(len(hyp_words) + 1):
		d[0, j] = j  # insertions

	# Fill the distance matrix
	for i in range(1, len(ref_words) + 1):
		for j in range(1, len(hyp_words) + 1):
			if ref_words[i-1] == hyp_words[j-1]:
				d[i, j] = d[i-1, j-1]  # Match, no cost
			else:
				# Take minimum of three operations
				substitution = d[i-1, j-1] + 1
				insertion = d[i, j-1] + 1
				deletion = d[i-1, j] + 1
				d[i, j] = min(substitution, insertion, deletion)

	# Backtrack to count each operation type
	i, j = len(ref_words), len(hyp_words)
	substitutions, insertions, deletions = 0, 0, 0

	while i > 0 or j > 0:
		if i > 0 and j > 0 and ref_words[i-1] == hyp_words[j-1]:
			i, j = i-1, j-1
		elif i > 0 and j > 0 and d[i, j] == d[i-1, j-1] + 1:
			substitutions += 1
			i, j = i-1, j-1
		elif j > 0 and d[i, j] == d[i, j-1] + 1:
			insertions += 1
			j = j-1
		elif i > 0 and d[i, j] == d[i-1, j] + 1:
			deletions += 1
			i = i-1
		else:
			break

	# Calculate metrics
	ref_len = len(ref_words)
	total_errors = substitutions + insertions + deletions
	wer = total_errors / ref_len if ref_len > 0 else float('inf')

	return {
		'wer': wer, 'substitutions': substitutions, 'insertions': insertions,
		'deletions': deletions, 'ref_length': ref_len, 'hyp_length': len(hyp_words),
		'total_errors': total_errors
	}


def find_optimal_speaker_assignment(gt_speakers_data, hyp_speakers_data):
	"""
	Find optimal speaker assignment for cpWER calculation.
	Uses Hungarian algorithm if available, otherwise greedy assignment.
	"""
	gt_speakers = list(gt_speakers_data.keys())
	hyp_speakers = list(hyp_speakers_data.keys())

	if not gt_speakers or not hyp_speakers:
		return [], {}

	cost_matrix = {}
	for gt_speaker in gt_speakers:
		cost_matrix[gt_speaker] = {}
		for hyp_speaker in hyp_speakers:
			gt_text = gt_speakers_data[gt_speaker]
			hyp_text = hyp_speakers_data.get(hyp_speaker, "") # Handle missing speaker in hypothesis
			wer_result = calculate_wer(gt_text, hyp_text)
			cost_matrix[gt_speaker][hyp_speaker] = wer_result['wer']

	try:
		from scipy.optimize import linear_sum_assignment
		num_cost_matrix = np.array([[cost_matrix[gt][hyp] for hyp in hyp_speakers] for gt in gt_speakers])
		row_ind, col_ind = linear_sum_assignment(num_cost_matrix)
		assignments = [(gt_speakers[r], hyp_speakers[c]) for r, c in zip(row_ind, col_ind)]
	except ImportError:
		logger.warning("scipy not available, using greedy assignment for cpWER")
		assignments = []
		used_hyp = set()
		for gt_speaker in sorted(gt_speakers, key=lambda g: min(cost_matrix[g].values())):
			best_hyp, min_cost = None, float('inf')
			for hyp_speaker, cost in cost_matrix[gt_speaker].items():
				if hyp_speaker not in used_hyp and cost < min_cost:
					min_cost, best_hyp = cost, hyp_speaker
			if best_hyp:
				assignments.append((gt_speaker, best_hyp))
				used_hyp.add(best_hyp)

	return assignments, cost_matrix


def calculate_cpwer(gt_speakers_data, hyp_speakers_data):
	"""
	Calculate concatenated minimum-permutation Word Error Rate (cpWER).
	"""
	if not gt_speakers_data:
		return {'cpwer': float('inf'), 'assignments': [], 'speaker_wers': {},
				'total_ref_words': 0, 'total_errors': sum(len(t.split()) for t in hyp_speakers_data.values())}

	assignments, cost_matrix = find_optimal_speaker_assignment(gt_speakers_data, hyp_speakers_data)
	total_ref_words, total_errors = 0, 0
	speaker_wers = {}

	# Matched speakers
	hyp_speakers_assigned = set()
	for gt_speaker, hyp_speaker in assignments:
		gt_text = gt_speakers_data[gt_speaker]
		hyp_text = hyp_speakers_data[hyp_speaker]
		wer_result = calculate_wer(gt_text, hyp_text)
		speaker_wers[f"{gt_speaker} -> {hyp_speaker}"] = wer_result
		total_ref_words += wer_result['ref_length']
		total_errors += wer_result['total_errors']
		hyp_speakers_assigned.add(hyp_speaker)

	# Unmatched GT speakers (deletions)
	for gt_speaker in set(gt_speakers_data.keys()) - {g for g, h in assignments}:
		ref_len = len(gt_speakers_data[gt_speaker].split())
		total_ref_words += ref_len
		total_errors += ref_len
		speaker_wers[f"{gt_speaker} -> UNMATCHED"] = {'wer': 1.0, 'total_errors': ref_len, 'ref_length': ref_len}

	# Unmatched hypothesis speakers (insertions)
	for hyp_speaker in set(hyp_speakers_data.keys()) - hyp_speakers_assigned:
		hyp_len = len(hyp_speakers_data[hyp_speaker].split())
		total_errors += hyp_len
		speaker_wers[f"UNMATCHED -> {hyp_speaker}"] = {'wer': float('inf'), 'total_errors': hyp_len, 'ref_length': 0}

	cpwer = total_errors / total_ref_words if total_ref_words > 0 else float('inf')

	return {
		'cpwer': cpwer, 'assignments': assignments, 'speaker_wers': speaker_wers,
		'total_ref_words': total_ref_words, 'total_errors': total_errors,
		'cost_matrix': cost_matrix
	}

def timestamp_to_seconds(ts_str):
    """Converts HH:MM:SS timestamp string to seconds."""
    parts = ts_str.split(':')
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

def load_ground_truth(gt_filepath):
	logger.info("Reading GT metadata and importing ground truth data (deduplicating utterances)...")
	transcripts = {}
	processed_utterances = set() # Keep track of (transcript_id, utterance_id) pairs
	
	try:
		with open(gt_filepath, 'r', encoding='utf-8') as file:
			csv_reader = csv.DictReader(file)
			
			# Store rows to process after finding the first utterance for each transcript
			rows = list(csv_reader)

			for row in rows:
				transcript_id = row.get('transcript_id')
				utterance_id_str = row.get('utterance_id')

				if not transcript_id or not utterance_id_str:
					continue
				
				# Initialize transcript entry if new
				if transcript_id not in transcripts:
					transcripts[transcript_id] = {
						'transcript_id' : transcript_id,
						'video_title' : row.get('video_title', 'N/A'),
						'video_url' : row.get('video_url', 'N/A'),
						'mi_quality' : row.get('mi_quality', 'N/A'),
						'all_utterances' : [],
						'who_said_what' : {},
						'start_time_seconds': 0.0 # Default start time
					}
				
				# *** NEW: Capture the start time from the first utterance ***
				if utterance_id_str == '0':
					timestamp = row.get('timestamp')
					if timestamp:
						transcripts[transcript_id]['start_time_seconds'] = timestamp_to_seconds(timestamp)
						logger.info(f"Transcript {transcript_id} has a start time of {transcripts[transcript_id]['start_time_seconds']}s")

				# --- Deduplication and data aggregation ---
				utterance_key = (transcript_id, int(utterance_id_str))
				if utterance_key in processed_utterances:
					continue
				processed_utterances.add(utterance_key)

				speaker = row.get('interlocutor', 'Unknown')
				text = row.get('utterance_text', '').strip()

				transcripts[transcript_id]['all_utterances'].append(text)
				if speaker not in transcripts[transcript_id]['who_said_what']:
					transcripts[transcript_id]['who_said_what'][speaker] = []
				transcripts[transcript_id]['who_said_what'][speaker].append(text)

	except FileNotFoundError:
		logger.error(f"Error: File not found at {gt_filepath}")
		return {}
	except Exception as e:
		logger.error(f"Error opening or reading file {gt_filepath}: {e}")
		return {}

	# Consolidate texts
	logger.info("Consolidating transcript texts...")
	for transcript_id, item in transcripts.items():
		item['transcript'] = ' '.join(item['all_utterances'])
		for speaker, their_texts in item['who_said_what'].items():
			item['who_said_what'][speaker] = ' '.join(their_texts)

	logger.info("Done importing and deduplicating ground truth data.")
	return transcripts

def count_speaker_accuracy(gt_num_speakers, hyp_num_speakers):
    return {
        'gt_speaker_count': gt_num_speakers,
        'hyp_speaker_count': hyp_num_speakers,
        'speaker_count_error': abs(gt_num_speakers - hyp_num_speakers),
        'speaker_count_correct': gt_num_speakers == hyp_num_speakers,
        'speaker_count_error_rate': abs(gt_num_speakers - hyp_num_speakers) / max(gt_num_speakers, 1)
    }

def write_results_to_file(results, output_filepath='out/evaluation_results.json'):
	os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
	with open(output_filepath, 'w', encoding='utf-8') as f:
		json.dump(results, f, indent=4)
	logger.info(f"Results successfully written to {output_filepath}")

def append_result_to_file(transcript_id, evaluation, output_filepath='out/evaluation_results_with_wer_cpwer.json'):
	os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
	results = {}
	if os.path.exists(output_filepath) and os.path.getsize(output_filepath) > 0:
		with open(output_filepath, 'r', encoding='utf-8') as f:
			try:
				results = json.load(f)
			except json.JSONDecodeError:
				logger.warning(f"Could not parse existing results file at {output_filepath}. Starting fresh.")
	
	results[transcript_id] = evaluation
	with open(output_filepath, 'w', encoding='utf-8') as f:
		json.dump(results, f, indent=4)

def trim_audio(input_path, output_path, start_sec):
    """Trims audio file from start_sec to the end and saves as WAV."""
    logger.info(f"Trimming audio file '{input_path}' from {start_sec} seconds.")
    try:
        # Using pydub for robust format handling, then librosa for processing
        audio = AudioSegment.from_file(input_path)
        # pydub uses milliseconds
        trimmed_audio = audio[start_sec * 1000:]
        # Export as a temporary WAV file for librosa/soundfile
        trimmed_audio.export(output_path, format="wav")
        logger.info(f"Trimmed audio saved to '{output_path}'")
        return True
    except Exception as e:
        logger.error(f"Failed to trim audio file {input_path}: {e}")
        return False

def main():
	logger.info("starting ...")
	comparison_results = {}
	output_filepath = 'out/evaluation_results_with_wer_cpwer.json'
	os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
	with open(output_filepath, 'w') as f: json.dump({}, f)

	accent_detector = AccentDetector(device=get_device())
	ground_truth = load_ground_truth('evaluation_data/AnnoMI-full-export-ground-truth.csv')

	transcripts_to_sample = list(ground_truth.keys())
	random.seed(42)
	num_to_sample = int(len(transcripts_to_sample) * 0.33)
	sampled_transcripts = random.sample(transcripts_to_sample, num_to_sample)
	logger.info(f"Transcripts to be sampled: {sampled_transcripts}")

	missing_audio = {"89"}
	# has_narrator = {"0", "1", "2", "7", "15", "16", "24", "34", "44", "63", "67", "72", "99", "106", "109", "112", "123", "129"}
	bad_annotation = {"3", "11", "14"}
	already_reviewed = {"4", "5", "8", "9", "10", "12", "13", "17"}

	# *** MODIFIED: No longer filtering out videos with narrators ***
	filtered_transcripts = {
		str(tid) for tid in sampled_transcripts
		if str(tid) not in missing_audio
		and str(tid) not in bad_annotation
	}
	logger.info(f"Total transcripts to process after filtering: {len(filtered_transcripts)}")
	approved_set = filtered_transcripts & already_reviewed
	uncategorized_set = filtered_transcripts - already_reviewed

	# log them
	logger.info(f"Approved ({len(approved_set)}): {approved_set.sorted()}")
	logger.info(f"Uncategorized (to be reviewed) ({len(uncategorized_set)}): {sorted(uncategorized_set)}")


	#
	# PROCESS THE LIST OF FILES
	#
	audio_filepaths = s3_download_files(filtered_transcripts)
	transcripts_speaker_count_errors = []

	for i, transcript_id in enumerate(filtered_transcripts, 1):
		logger.info(f"--- Processing transcript {transcript_id} ({i}/{len(filtered_transcripts)}) ---")

		audio_info = audio_filepaths[transcript_id]
		original_audio_filepath = audio_info['filepath']
		audio_duration = audio_info['duration']
		
		transcript_info = ground_truth[transcript_id]
		title = transcript_info['video_title']
		start_time = transcript_info.get('start_time_seconds', 0.0)
		
		evaluation = {
			'audio_duration': audio_duration,
			'audio_duration_minutes': audio_duration / 60.0,
			'gt_speakers': list(transcript_info['who_said_what'].keys()),
			'gt_num_speakers': len(transcript_info['who_said_what'].keys()),
			'downloaded_filepath': original_audio_filepath,
			'has_narrator': str(transcript_id) in has_narrator,
			'processing_start_time_sec': start_time
		}

		# Accent Detection (runs on the full original audio)
		try:
			detected_accents = accent_detector.detect_accent(original_audio_filepath)
			evaluation['accent_detection'] = detected_accents
			evaluation['model_accent'] = detected_accents.get('primary_prediction', 'Unknown')
		except Exception as e:
			logger.warning(f"Accent detection failed for {transcript_id}: {e}")
			evaluation['accent_detection'] = {"error": str(e)}
			evaluation['model_accent'] = 'Unknown'

		gt_transcript = transcript_info['transcript']
		gt_wsw_transcript = transcript_info['who_said_what']
		evaluation['gt_word_count'] = len(gt_transcript.split())
		evaluation['gt_speaker_word_counts'] = {s: len(t.split()) for s, t in gt_wsw_transcript.items()}

		# *** NEW: Trim audio if a start time is specified ***
		processing_filepath = original_audio_filepath
		temp_filepath = None

		if start_time > 0:
			temp_filepath = f"out/temp_{transcript_id}_trimmed.wav"
			if trim_audio(original_audio_filepath, temp_filepath, start_time):
				processing_filepath = temp_filepath
			else:
				logger.error(f"Could not trim audio for {transcript_id}. Skipping processing.")
				continue
		
		try:
			results = process_audio(processing_filepath)

			for processor, proc_data in results.items():
				logger.info(f"Evaluating GT against processor: {processor}")
				evaluation[processor] = {}

				# WER on continuous transcript
				processor_transcript = proc_data['continuous_transcript']
				wer_result = calculate_wer(gt_transcript, processor_transcript)
				evaluation[processor]['wer_metrics'] = wer_result

				# Speaker diarization and cpWER
				processor_wsw = proc_data['whosaidwhat_transcript']
				num_hyp_speakers = len(processor_wsw.keys())
				speaker_count_metrics = count_speaker_accuracy(evaluation['gt_num_speakers'], num_hyp_speakers)
				evaluation[processor]['speaker_count_metrics'] = speaker_count_metrics
				if not speaker_count_metrics['speaker_count_correct']:
					transcripts_speaker_count_errors.append({'transcript_id': transcript_id, 'processor': processor, 'gt': evaluation['gt_num_speakers'], 'hyp': num_hyp_speakers})

				cpwer_result = calculate_cpwer(gt_wsw_transcript, processor_wsw)
				evaluation[processor]['cpwer_metrics'] = cpwer_result
				evaluation[processor]['individual_speaker_wers'] = {k: v['wer'] for k, v in cpwer_result.get('speaker_wers', {}).items()}

			append_result_to_file(transcript_id, evaluation, output_filepath)
			comparison_results[transcript_id] = evaluation
			logger.info(f"Evaluation for transcript {transcript_id} completed and saved.")

		finally:
			# *** NEW: Clean up the temporary file ***
			if temp_filepath and os.path.exists(temp_filepath):
				os.remove(temp_filepath)
				logger.info(f"Removed temporary file: {temp_filepath}")

	logger.info("Workflow finished.")
	logger.info(f"All results have been written to {output_filepath}")

	if transcripts_speaker_count_errors:
		logger.info("Transcripts with speaker count errors:")
		for item in transcripts_speaker_count_errors:
			logger.info(f"  {item['transcript_id']}: {item['processor']} detected {item['hyp']} vs GT {item['gt']}")
	
	generate_summary_statistics(output_filepath)

# The rest of your script (generate_summary_statistics, etc.) remains the same.
# ...
if __name__ == "__main__":
	main()