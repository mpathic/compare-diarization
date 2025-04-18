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