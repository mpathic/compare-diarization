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

HF_TOKEN = os.getenv('HF_TOKEN', None)

whisp_model = 'medium'
diarize_model = "pyannote/speaker-diarization-3.1"
vad_model = "pyannote/voice-activity-detection"

# Load the Whisper model for transcription
logger.info(f"Loading Whisper model: {whisp_model}")

print(whisper.__version__)

whisper_model = whisper.load_model(whisp_model)


def process(audio_file):

	# make outfile names
	os.makedirs('out', exist_ok = True)
	basename = os.path.basename(audio_file)









	# format output response
	unified_transcript, what_speakers_said = format_results(transcript)

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

	logger.debug("Open Source process ...")
