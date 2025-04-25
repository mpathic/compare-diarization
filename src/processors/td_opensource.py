import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import Counter

import torch
import whisper
import numpy as np
import librosa
from pyannote.audio import Pipeline, Audio
from pyannote.core import Segment, Timeline, Annotation


logger = logging.getLogger(__name__) # respect mains loglevel


def log_vad_results(vad_result, audio_duration):
	"""Log detailed information about VAD results.
	"""
	logger.info("=== DETAILED VAD RESULTS ===")
	
	# Extract speech segments with scores
	speech_segments = []
	for segment, _, score in vad_result.itertracks(yield_label=True):
		speech_segments.append({
			"start": segment.start,
			"end": segment.end,
			"duration": segment.end - segment.start
		})
	
	# Sort segments by start time
	speech_segments.sort(key=lambda x: x["start"])
	
	# Calculate statistics
	total_speech_time = sum(seg["duration"] for seg in speech_segments)
	total_silence_time = audio_duration - total_speech_time
	speech_percentage = (total_speech_time / audio_duration) * 100
	silence_percentage = (total_silence_time / audio_duration) * 100
	
	# Log statistics
	logger.info(f"Found {len(speech_segments)} speech segments")
	logger.info(f"Total speech time: {total_speech_time:.2f}s ({speech_percentage:.1f}% of audio)")
	logger.info(f"Total silence time: {total_silence_time:.2f}s ({silence_percentage:.1f}% of audio)")
	
	# Log segment distribution
	segment_durations = [seg["duration"] for seg in speech_segments]
	if segment_durations:
		avg_duration = sum(segment_durations) / len(segment_durations)
		max_duration = max(segment_durations)
		min_duration = min(segment_durations)
		
		logger.debug(f"Segment statistics: avg={avg_duration:.2f}s, min={min_duration:.2f}s, max={max_duration:.2f}s")
	
	# Log individual segments for debugging
	if logger.level <= logging.DEBUG:
		for i, segment in enumerate(speech_segments):
			logger.debug(
				f"  Segment {i+1}: {segment['start']:.2f}s - {segment['end']:.2f}s " +
				f"({segment['duration']:.2f}s, score=NA)"
			)
	
	# Analyze gaps between segments
	gaps = []
	for i in range(len(speech_segments) - 1):
		gap_start = speech_segments[i]["end"]
		gap_end = speech_segments[i+1]["start"]
		gap_duration = gap_end - gap_start
		
		if gap_duration > 0.1:  # Only log meaningful gaps
			gaps.append({
				"start": gap_start,
				"end": gap_end,
				"duration": gap_duration
			})
	
	# Log gap statistics
	total_gap_time = sum(gap["duration"] for gap in gaps)
	logger.debug(f"Found {len(gaps)} gaps between speech segments, total gap time: {total_gap_time:.2f}s")
	
	for gap in gaps:
		logger.debug(f" gap start:{gap['start']:.3f} end:{gap['end']:.3f} duration:{gap['duration']:.3f}")
	
	# Log potential issues
	long_segments = [seg for seg in speech_segments if seg["duration"] > 10.0]
	short_segments = [seg for seg in speech_segments if seg["duration"] < 0.3]
	
	if long_segments:
		logger.warning(f"Found {len(long_segments)} unusually long segments (>10s)")
		for seg in long_segments:
			logger.warning(f"  Long segment: {seg['start']:.2f}s - {seg['end']:.2f}s ({seg['duration']:.2f}s)")
	
	if short_segments:
		logger.warning(f"Found {len(short_segments)} very short segments (<0.3s)")
	
	logger.debug("=== END VAD RESULTS ===")
	
	# Return summary for saving to output
	return {
		"speech_segments": speech_segments,
		"total_speech_time": total_speech_time,
		"total_silence_time": total_silence_time,
		"speech_percentage": speech_percentage,
		"silence_percentage": silence_percentage,
		"segment_stats": {
			"count": len(speech_segments),
			"avg_duration": avg_duration if segment_durations else 0,
			"min_duration": min_duration if segment_durations else 0,
			"max_duration": max_duration if segment_durations else 0
		},
		"gaps": gaps,
		"total_gap_time": total_gap_time
	}


def log_diarization_results(diarization_result):
	"""Log detailed information about diarization results.
	"""
	logger.info("=== DETAILED DIARIZATION RESULTS ===")
	
	# Extract speaker turns with detailed timing
	ordered_turns = []
	speaker_turns = {}
	speaker_total_time = {}
	
	for turn, _, speaker in diarization_result.itertracks(yield_label=True):
		if speaker not in speaker_turns:
			speaker_turns[speaker] = []
			speaker_total_time[speaker] = 0
		
		duration = turn.end - turn.start
		item = {
			"speaker" : speaker,
			"start": turn.start,
			"end": turn.end,
			"duration": duration
		}
		speaker_turns[speaker].append(item)
		ordered_turns.append(item)
		logger.info(f"{item['speaker']}: ({item['start']:.3f} - {item['end']:.3f}) duration [{item['duration']:.3f}]")

		speaker_total_time[speaker] += duration
		
	# Log detailed speaker information
	for speaker in sorted(speaker_turns.keys()):
		turns = speaker_turns[speaker]
		total_time = speaker_total_time[speaker]
		avg_turn_length = total_time / len(turns) if turns else 0
		
		logger.info(f"Speaker {speaker}: {len(turns)} turns, {total_time:.2f}s total time, {avg_turn_length:.2f}s avg turn length")
		
		# Log individual turns for debugging
		if logger.level <= logging.DEBUG:
			for i, turn in enumerate(turns):
				logger.debug(f"  Turn {i+1}: {turn['start']:.2f}s - {turn['end']:.2f}s ({turn['duration']:.2f}s)")
	
	# Calculate overlap statistics
	overlaps = []
	all_turns = []
	for speaker, turns in speaker_turns.items():
		for turn in turns:
			all_turns.append({"speaker": speaker, "start": turn["start"], "end": turn["end"]})
	
	# Sort by start time
	all_turns.sort(key=lambda x: x["start"])
	
	# Check for overlaps
	for i in range(len(all_turns) - 1):
		current = all_turns[i]
		next_turn = all_turns[i+1]
			
		if current["end"] > next_turn["start"]:
			overlap = {
				"speaker1": current["speaker"],
				"speaker2": next_turn["speaker"],
				"start": next_turn["start"],
				"end": min(current["end"], next_turn["end"]),
				"duration": min(current["end"], next_turn["end"]) - next_turn["start"]
			}
			overlaps.append(overlap)
	
	# Log overlap statistics
	total_overlap_time = sum(o["duration"] for o in overlaps)
	logger.info(f"Found {len(overlaps)} speaker overlaps, {total_overlap_time:.2f}s total overlap time")
	
	# Log individual overlaps for debugging
	if logger.level <= logging.DEBUG and overlaps:
		for i, overlap in enumerate(overlaps):
			logger.debug(
				f"  Overlap {i+1}: {overlap['speaker1']} and {overlap['speaker2']} " +
				f"from {overlap['start']:.2f}s to {overlap['end']:.2f}s ({overlap['duration']:.2f}s)"
			)
	
	logger.info("=== END DIARIZATION RESULTS ===")
	
	# Return summary for saving to output
	return {
		"speakers": list(speaker_turns.keys()),
		"speaker_turns": speaker_turns,
		"speaker_total_time": speaker_total_time,
		"overlaps": overlaps,
		"total_overlap_time": total_overlap_time,
		"ordered_speaker_turns" : ordered_turns
	}


def extract_audio_segment(
		audio_path: str,
		start_time: float,
		end_time: float
	) -> np.ndarray:
	"""Extract audio segment from file.
	
	Args:
		audio_path: Path to audio file
		start_time: Start time in seconds
		end_time: End time in seconds
		
	Returns:
		Audio segment as numpy array
	"""
	# Load full audio
	audio, sr = librosa.load(audio_path, sr=16000)
	
	# Convert times to samples
	start_sample = int(start_time * sr)
	end_sample = int(end_time * sr)
	
	# Extract segment
	segment = audio[start_sample:end_sample]
	
	return segment


def join_diarization_transcription(speech_regions, audio_path, whisper_model):
	# Now, get the transcription according to speech segments
	# noticed better performance processing in these chunks
	# transcribe each segment
	logger.info("Combining clipping audio and transcribing based on speech regions ...")
	utterance_textlist = []
	utterances_with_details = []
	
	# either speech_regions OR adjusted_segments, try both
	for i, segment in enumerate(speech_regions):

		logger.info(f"Transcribing segment {i+1}/{len(speech_regions)}: {segment['start']:.2f}s - {segment['end']:.2f}s")
		
		# Extract audio segment
		audio_segment = extract_audio_segment(
			audio_path, segment["start"], segment["end"]
		)
		
		# Skip empty segments
		if len(audio_segment) == 0:
			logger.warning(f"Segment {i+1} is empty, skipping")
			continue
			
		#
		# Transcribe with Whisper
		#
		segment_transcription = whisper_model.transcribe(
			audio_segment,
			condition_on_previous_text=True,
			temperature=0.0,
			compression_ratio_threshold=1.35,
			verbose=True,
			language="en"  # specify English language for now, add detect later, inconsistent between chunks sometimes
		)
		
		# Log whisper segments
		logger.debug(f"Segment {i+1}: Got {len(segment_transcription['segments'])} Whisper segments")
		logger.debug(segment_transcription['text'])

		global_start = segment["start"]
		global_end = segment["end"]

		j=1
		for local_segment in segment_transcription['segments']:
			local_start = local_segment["start"]
			local_end = local_segment["end"]
			local_text = local_segment['text'].strip()

			utterances_with_details.append({
				'segment' : i,
				'global_segment_start' : global_start,
				'global_segment_end' : global_end,
				'local_subsegment' : j,
				'local_start' : local_start,
				'local_end' : local_end,
				'text' : local_text
			})
			utterance_textlist.append(local_text)
		
			logger.debug(f"\tSub-Seg {j}, global({global_start:.3f}-{global_end:.3f}) local({local_start:.2f}-{local_end:.2f}) text({local_text})")
			j+=1

	return utterance_textlist, utterances_with_details


def write_basic_output_to_disk(audio_path, results):
	# make outfile name
	os.makedirs('out', exist_ok = True)
	basename = os.path.basename(audio_path)
	outfile = f"out/out_opensource_basic_{basename}.txt"
	with open(outfile, 'w', encoding='utf-8') as f:

		for i in results:
			string = f"({i['utterance_segment_id']}:{i['utterance_subsegment_id']}) ({i['utterance_start']:.3f}-{i['utterance_end']:.3f}) ({i['assigned_speaker']}) ({i['utterance_text']})"
			logger.debug(string)
			f.write(f"{string}\n")

def write_diarized_output_to_disk(audio_path, diarized_group):
	basename = os.path.basename(audio_path)
	outfile = f"out/out_opensource_diarized_group_{basename}.txt"
	with open(outfile, 'w', encoding='utf-8') as f:

		for speaker, texts in diarized_group.items():
			f.write(f"{speaker}:\n{texts}\n")

def write_transcript_to_disk(audio_path, transcript):
	basename = os.path.basename(audio_path)
	outfile = f"out/out_opensource_transript_{basename}.txt"
	with open(outfile, 'w', encoding='utf-8') as f:
		f.write(f"{transcript}")


def process(audio_path):
	logger.info("Starting Open Source method")
	#
	# Setup
	# 
	whisp_model = 'medium'
	diarize_model = "pyannote/speaker-diarization-3.1"
	vad_model = "pyannote/voice-activity-detection"

	# make outfile names
	os.makedirs('out', exist_ok = True)
	basename = os.path.basename(audio_path)

	# Load the Whisper model for transcription
	logger.info(f"Loading Whisper model size: {whisp_model}")
	logger.debug(f"whisper version: {whisper.__version__}")
	whisper_model = whisper.load_model(whisp_model)

	# load diarization
	logger.debug(f"Loading Diarization model: {diarize_model}")
	diarization_model = Pipeline.from_pretrained(diarize_model, use_auth_token=os.environ.get('HF_TOKEN'))

	# load vad
	logger.debug(f"Loading Diarization model: {vad_model}")
	vad_model = Pipeline.from_pretrained(vad_model, use_auth_token=os.environ.get('HF_TOKEN'))

	# get some basic stats about the file
	logger.info(f"Processing audio file: {audio_path}")
	audio = Audio() # init the audio analyzer

	audio_duration = audio.get_duration(audio_path)
	logger.info(f"Total audio duration: {audio_duration:.2f} seconds")

	#
	# VAD model
	#
	# # # RUN file against VAD model
	logger.info("Running voice activity detection ...")
	vad_result = vad_model(audio_path)
	vad_analysis = log_vad_results(vad_result, audio_duration)

	# Get speech segments from VAD
	min_segment_duration = 0.25
	speech_regions = []

	for segment, track, score in vad_result.itertracks(yield_label=True):
		duration = segment.end - segment.start
		if duration >= min_segment_duration:
			speech_regions.append({
				"start": segment.start,
				"end": segment.end
			})

	for item in speech_regions:
		logger.debug(f"speech region ==> {item['start']:.3f} - {item['end']:.3f}")

	logger.info(f"Processing {len(speech_regions)} segments for transcription")

	#
	# Diarization model
	#
	# Run diarization
	logger.info("Running speaker diarization ...")
	diarization_result = diarization_model(audio_path)
	diarization_analysis = log_diarization_results(diarization_result)

	#
	# The logic to tie them together
	#
	utterance_textlist, utterances = join_diarization_transcription(speech_regions, audio_path, whisper_model)
	# utterance_textlist, utterances = join_diarization_transcription(adjusted_segments)

	# now, iterate over the utterances, determine the absolute timestamps for each text
	turns = diarization_analysis['ordered_speaker_turns']

	results = []
	for utterance in utterances:
		
		id_global = utterance['segment']
		id_local = utterance['local_subsegment']
		utt_start_time = utterance['global_segment_start'] + utterance['local_start']
		utt_end_time = utterance['global_segment_start'] + utterance['local_end']
		utt_text = utterance['text']

		logger.debug(f"({id_global}:{id_local}) ({utt_start_time:.3f}-{utt_end_time:.3f}) ({utterance['text']})")

		# now go thru the diarization speech turns
		max_overlap = 0.0
		max_overlap_speaker = None
		max_overlap_turn_details = None

		k=0
		for turn in turns:
			k+=1
			turn_start = turn['start']
			turn_end = turn['end']
			turn_speaker = turn['speaker']

			# now calculate te overlapps
			overlap_start = max(utt_start_time, turn_start) # whichever is later on
			overlap_end = min(utt_end_time, turn_end) # whichever goes first
			# ^ a super short utterance may not be picked up by this, but if it's super short (yeah, mhm) maybe we don't care about that text anyways
			# so , it could struggle with backchannel speech

			overlap_duration = max(0, overlap_end - overlap_start)
			logger.debug(f"\tComparing with Turn {k}: ({turn_start:.3f}-{turn_end:.3f}) Speaker: {turn_speaker} -> Overlap: {overlap_duration:.3f}s")
			
			# now check if it should replace current candidate speaker
			if overlap_duration > max_overlap:
				max_overlap = overlap_duration
				max_overlap_speaker = turn_speaker
				max_overlap_turn_details = {
					'turn_index' : k,
					'turn_start' : turn_start,
					'turn_end' : turn_end,
					'turn_speaker' : turn_speaker,
					'overlap_duration' : overlap_duration
				}
				logger.debug("\t\t* new max (!)")

		# ok now, print the best match with most overlap for this utterance
		if max_overlap_speaker:
			logger.debug(f"\t Assigned Speaker: {max_overlap_speaker} (Max Overlap: {max_overlap:.3f} Turn: {max_overlap_turn_details})\n")
			results.append({
				'utterance_segment_id': id_global,
				'utterance_subsegment_id': id_local,
				'utterance_start': utt_start_time,
				'utterance_end': utt_end_time,
				'utterance_text': utt_text,
				'assigned_speaker': max_overlap_speaker,
				'max_overlap_duration': max_overlap,
				'matched_turn_details': max_overlap_turn_details
			})
		else:
			logger.debug("\tNo speaker overlap found for this utterance (!) \n")
			results.append({
				'utterance_segment_id': id_global,
				'utterance_subsegment_id': id_local,
				'utterance_start': utt_start_time,
				'utterance_end': utt_end_time,
				'utterance_text': utt_text,
				'assigned_speaker': 'UNKNOWN',
				'max_overlap_duration': 0.0,
				'matched_turn_details': None
		})

	# format results to pass the objects back to caller
	speaker_blocks = {}
	for res in results:
		speaker = res['assigned_speaker']
		# Skip UNKNOWN speakers entirely, usually just one word attributed to UNK
		# and i would rather have this detect the number of speakers more accurately
		# than adding an entirely new speaker for one word.
		if speaker == 'UNKNOWN':
			continue

		utt_text = res['utterance_text'].strip()

		if speaker not in speaker_blocks.keys():
			speaker_blocks[speaker] = []

		speaker_blocks[speaker].append(utt_text)


	# ignore the speaker if they only talk for less than a half second overakll
	# calculate total talk time for each speaker
	speaker_talk_times = {}
	for res in results:
		speaker = res['assigned_speaker']
		duration = res['utterance_end'] - res['utterance_start']
		
		if speaker not in speaker_talk_times:
			speaker_talk_times[speaker] = 0
		
		speaker_talk_times[speaker] += duration

	# filter out speakers with less than 0.5 sec of talk time
	speakers_to_ignore = [speaker for speaker, talk_time in speaker_talk_times.items() if talk_time < 0.5]
	logger.info(f"Ignoring {len(speakers_to_ignore)} speakers with less than 0.5 seconds of talk time: {speakers_to_ignore}")

	# create what_speakers_said but exclude ignored speakers
	what_speakers_said = {}
	for speaker, textlist in speaker_blocks.items():
		if speaker not in speakers_to_ignore:
			what_speakers_said[speaker] = ' '.join(textlist)
			# logger.debug(f"\t {speaker}:{what_speakers_said}")


	transcript_block = ' '.join(utterance_textlist)  # unified transcript

	# drop the results to disk
	write_basic_output_to_disk(audio_path, results)
	write_diarized_output_to_disk(audio_path, what_speakers_said)
	write_transcript_to_disk(audio_path, transcript_block)

	result = {
		'continuous_transcript' : transcript_block,
		'whosaidwhat_transcript' : what_speakers_said
	}
	return result


if __name__ == '__main__':

	logger.debug("Open Source process ...")
