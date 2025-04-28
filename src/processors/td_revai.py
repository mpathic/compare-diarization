import os
import json
import logging

from rev_ai import apiclient
from time import sleep

logger = logging.getLogger(__name__) # respect mains loglevel


def make_output(transcript):
	transcript_utterances = []

	for monologue in transcript['monologues']:

		speaker = monologue.get('speaker', 'Unknown Speaker')
		words = []
		start_ts = None
		end_ts = None

		for elem in monologue['elements']:
			# print(elem)

			if elem['type'] == 'text':
				words.append(elem['value'])
				if start_ts is None:
					start_ts = elem['ts']
				end_ts = elem['end_ts']
			else:
				words.append(elem['value'])

		utterance_text = ''.join(words)
		string = (f"[{start_ts:.2f} - {end_ts:.2f}] Speaker_{speaker}: {utterance_text}")
		logger.debug(string)

		transcript_utterances.append({
			'start_ts' : start_ts,
			'end_ts' : end_ts,
			'speaker' : speaker,
			'text' : utterance_text
			})

	return transcript_utterances


def save_revai_basic_output(transcript_utt, output_path='revai_basic_transcript.txt'):
	with open(output_path, 'w', encoding='utf-8') as f:
		for utterance in transcript_utt:
			speaker = utterance['speaker']
			text = utterance['text'].strip()
			start_ts = utterance['start_ts']
			end_ts = utterance['end_ts']

			string = (f"[{start_ts:.2f} - {end_ts:.2f}] Speaker_{speaker}: {text}")
			f.write(f"{string}\n")


def save_revai_transcript_output(transcript, output_path='revai_trans_transcript.txt'):
	with open(output_path, 'w', encoding='utf-8') as f:
		for utterance in transcript:
			text = utterance['text'].strip()
			f.write(f"{text} ") # add a space after, its fine


def save_revai_diarized_output(transcript, output_path='revai_diarized_transcript.txt'):
	speaker_texts = {}

	for utt in transcript:
		if utt['speaker'] not in speaker_texts.keys():
			speaker_texts[utt['speaker']] = []

		speaker_texts[utt['speaker']].append(utt['text'].strip())

	with open(output_path, 'w', encoding='utf-8') as f:
		for speaker in speaker_texts.keys():
			# join their texts together with a space
			speakers_utterances = ' '.join(speaker_texts[speaker])
			f.write(f"{speaker}:\n{speakers_utterances}\n")


def write_outputs(transcript_utterances, basename):
	os.makedirs('out', exist_ok = True)
	outfile_basic = f"out/{basename}_revai_basic.txt"
	outfile_transcript = f"out/{basename}_revai_transcript.txt"
	outfile_diarized = f"out/{basename}_revai_diarized_group.txt"

	save_revai_basic_output(transcript_utterances, outfile_basic)
	save_revai_transcript_output(transcript_utterances, outfile_transcript)
	save_revai_diarized_output(transcript_utterances, outfile_diarized)


def format_result(transcript):
	speaker_texts = {}
	continuous_transcription = []

	for utt in transcript:
		continuous_transcription.append(utt['text'].strip()) # continuous

		if utt['speaker'] not in speaker_texts.keys():
			speaker_texts[utt['speaker']] = []
		speaker_texts[utt['speaker']].append(utt['text'].strip()) # diarized

	# join them into a long string
	unified_transcript = ' '.join(continuous_transcription)

	what_speakers_said = {}
	for speaker, textlist in speaker_texts.items():
		what_speakers_said[speaker] = ' '.join(textlist)

	return unified_transcript, what_speakers_said


def callback(transcript, filepath):

	# initial collection of the words
	transcript_utterances = make_output(transcript) # format the results

	# write the results to disk
	basename = os.path.basename(filepath)
	write_outputs(transcript_utterances, basename)

	#  now parse the objects for the return
	unified_transcript, what_speakers_said = format_result(transcript_utterances)

	logger.debug(f"Continuous transcription: {unified_transcript}")
	logger.debug(f"Diarized who said what text:")
	for k,v in what_speakers_said.items():
		logger.debug(f"\t {k}:{v}")

	result = {
		'continuous_transcript' : unified_transcript,
		'whosaidwhat_transcript' : what_speakers_said
	}
	return result


def process(audio_file):
	logger.info("Starting Rev AI method")

	# initialize Rev AI API client
	REVAI_TOKEN = os.environ.get('REVAI_TOKEN')
	client = apiclient.RevAiAPIClient(REVAI_TOKEN)

	# submit a file for transcription
	job = client.submit_job_local_file(audio_file)

	# get job id
	job_id = job.id
	logger.debug("Job submitted with id: " + job_id)

	# check job status
	while (job.status.name == 'IN_PROGRESS'):

		details = client.get_job_details(job_id)
		logger.debug("Job status: " + details.status.name)

		# if successful, print result
		if (details.status.name == 'TRANSCRIBED'):

			transcript = client.get_transcript_json(job_id)
			results = callback(transcript, audio_file)
			return results
			break

		# if unsuccessful, print error
		if (details.status.name == 'FAILED'):
			logger.error("Rev AI Job failed: " + details.failure_detail)
			break

		sleep(30)



if __name__ == '__main__':

	logger.debug("REV AI ...")

	# full_transcript, diarized_group_transcript = process(audio_file)


