import os
import json
import logging 

import assemblyai as aai

logger = logging.getLogger(__name__) # respect mains loglevel


def save_assemblyai_basic_output(transcript, output_path='assemblyai_basic_transcript.txt'):
	with open(output_path, 'w', encoding='utf-8') as f:
		for utterance in transcript.utterances:
			start = utterance.start / 1000  # ms to seconds
			end = utterance.end / 1000
			speaker = utterance.speaker
			text = utterance.text.strip()

			string = f"[{start:.2f} - {end:.2f}] Speaker_{speaker}: {text}"
			logger.debug(string)
			f.write(f"{string}\n")

def save_assemblyai_transcript_output(transcript, output_path='assemblyai_trans_transcript.txt'):
	with open(output_path, 'w', encoding='utf-8') as f:
		for utterance in transcript.utterances:
			text = utterance.text.strip()
			f.write(f"{text}")

def save_assemblyai_diarized_output(transcript, output_path='assemblyai_diarized_transcript.txt'):
	speaker_texts = {}
	for utt in transcript.utterances:
		if utt.speaker not in speaker_texts.keys():
			speaker_texts[utt.speaker] = []

		speaker_texts[utt.speaker].append(utt.text.strip())

	with open(output_path, 'w', encoding='utf-8') as f:
		for speaker in speaker_texts.keys():
			# join their texts together with nothing
			speakers_utterances = ' '.join(speaker_texts[speaker])
			f.write(f"{speakers_utterances}\n")



def format_result(transcript):
	speaker_texts = {}
	continuous_transcription = []

	for utt in transcript.utterances:
		continuous_transcription.append(utt.text.strip()) # continuous

		if utt.speaker not in speaker_texts.keys():
			speaker_texts[utt.speaker] = []
		speaker_texts[utt.speaker].append(utt.text.strip()) # diarized

	# join them into a long string
	unified_transcript = ' '.join(continuous_transcription)

	what_speakers_said = {}
	for speaker, textlist in speaker_texts.items():
		what_speakers_said[speaker] = ' '.join(textlist)

	return unified_transcript, what_speakers_said


def process(audio_file):

	# make outfile names
	os.makedirs('out', exist_ok = True)
	basename = os.path.basename(audio_file)
	outfile_basic = f"out/out_assemblyai_basic_{basename}.txt"
	outfile_transcript = f"out/out_assemblyai_transcript_{basename}.txt"
	outfile_diarized = f"out/out_assemblyai_diarized_group_{basename}.txt"

	aai.settings.api_key = os.environ.get('AAI_KEY')
	config = aai.TranscriptionConfig(
		speaker_labels=True
	)

	transcript = aai.Transcriber().transcribe(audio_file, config)

	save_assemblyai_basic_output(transcript, outfile_basic)
	save_assemblyai_transcript_output(transcript, outfile_transcript)
	save_assemblyai_diarized_output(transcript, outfile_diarized)

	# format output response
	unified_transcript, what_speakers_said = format_result(transcript)

	logger.debug(f"Continuous transcription: {unified_transcript}")
	logger.debug(f"Diarized who said what text:")
	for k,v in what_speakers_said.items():
		logger.debug(f"\t {k}:{v}")

	result = {
		'continuous_transcript' : unified_transcript,
		'split_diarized_transcript' : what_speakers_said
	}
	return result


if __name__ == '__main__':

	logger.debug("Assembly AI process ...")





