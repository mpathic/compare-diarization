# Speaker Diarization Evaluation with WER and cpWER

This project evaluates speaker diarization systems using standard Word Error Rate (WER) and concatenated minimum-permutation Word Error Rate (cpWER) metrics, with accent detection capabilities for comprehensive analysis.

## Key Changes from Original Code

### 1. Proper WER Implementation
- **Standard WER**: Implemented word-level Levenshtein distance calculation as per the original paper definition: `WER = (S + D + I) / N`
- **Removed WSWER**: Replaced the custom WSWER metric with industry-standard cpWER

### 2. cpWER Implementation  
- **cpWER (concatenated minimum-permutation WER)**: Finds optimal speaker assignment that minimizes overall WER
- **Hungarian Algorithm**: Uses scipy's `linear_sum_assignment` for optimal speaker matching when available
- **Handles speaker mismatches**: Properly accounts for unmatched speakers as deletions/insertions

### 3. Accent Detection
- **Hugging Face Integration**: Uses `Jzuluaga/accent-id-commonaccent_ecapa` model for accent detection
- **Multiple Accent Support**: Detects 16 different English accents including British, American, Southern US, etc.
- **Audio Segment Analysis**: Can analyze individual speaker segments for accent-specific evaluation

### 4. Poster Graphics Generation
- **Graphic 1**: Headline results comparing WER vs cpWER across systems
- **Graphic 2**: Duration cliff analysis showing performance degradation after 10 minutes
- **Graphic 3**: Speaker count accuracy highlighting system flaws
- **Graphic 4**: Accent robustness analysis showing bias in different models

## Requirements

```
# Core dependencies
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
torch>=1.9.0
torchaudio>=0.9.0
librosa>=0.8.1
soundfile>=0.10.3

# Speech processing
speechbrain>=0.5.12
transformers>=4.21.0
datasets>=2.0.0

# Evaluation metrics
python-Levenshtein>=0.12.0
editdistance>=0.6.0

# Data processing
python-dotenv>=0.19.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Optional: For advanced speaker assignment optimization
scipy>=1.7.0

# API clients (if using commercial services)
assemblyai>=0.15.0
# rev_ai  # Add if using RevAI

# File handling
boto3>=1.20.0  # If using S3 for audio storage
```

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/mpathic/compare-diarization
cd compare-diarization
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install SpeechBrain for accent detection:**
```bash
pip install speechbrain
```

5. **Set up environment variables:**
Create a `.env` file with your API keys:
```
AAI_KEY=your_assemblyai_api_key
REVAI_TOKEN=your_revai_token
HF_TOKEN=your_huggingface_token
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
```

## Usage

### 1. Basic Evaluation
```bash
python3 -m src.main
```

### 2. Generate Poster Graphics
```bash
python3 src/visualization_script.py
```

### 3. Test Accent Detection
```python
from src.accent_detector import AccentDetector

detector = AccentDetector()
result = detector.detect_accent("path/to/audio.wav")
print(f"Detected accent: {result['primary_prediction']}")
```

## File Structure

```
compare-diarization/
├── src/
│   ├── main.py                 # Main evaluation script (refactored)
│   ├── accent_detector.py      # New accent detection module  
│   ├── visualization_script.py # New graphics generation
│   ├── processors/
│   │   ├── td_assemblyai.py   # AssemblyAI processor
│   │   ├── td_revai.py        # RevAI processor
│   │   └── td_opensource.py   # Whisper + Pyannote processor
│   ├── s3_fh.py               # S3 file handling
│   └── download_videos.py     # Video download utilities
├── out/
│   ├── graphics/              # Generated poster graphics
│   ├── evaluation_results_with_wer_cpwer.json
│   └── summary_statistics.json
├── evaluation_data/
│   └── AnnoMI-full-export-ground-truth.csv
├── requirements.txt
└── README.md
```

## Key Metrics Explained

### Word Error Rate (WER)
**Formula**: `WER = (S + D + I) / N`
- **S**: Substitutions (wrong words)  
- **D**: Deletions (missed words)
- **I**: Insertions (extra words)
- **N**: Total reference words

**Example**: 
- Reference: "the quick brown fox"
- Hypothesis: "the slow brown dog"  
- WER = (2 substitutions + 0 deletions + 0 insertions) / 4 = 50%

### Concatenated Minimum-Permutation WER (cpWER)
**Purpose**: Evaluates both transcription accuracy AND speaker attribution

**Process**:
1. Find optimal assignment between ground truth speakers and hypothesis speakers
2. Calculate WER for each matched speaker pair
3. Aggregate errors across all speakers: `cpWER = Total_Errors / Total_Reference_Words`

**Why cpWER > WER**: cpWER accounts for speaker diarization errors, while WER ignores who said what

## Expected Outputs

### 1. JSON Results File
- Individual transcript evaluations
- WER and cpWER metrics for each system
- Speaker count accuracy
- Accent detection results
- Audio duration analysis

### 2. Poster Graphics
- **Graphic 1**: Overall performance comparison (WER vs cpWER)
- **Graphic 2**: Duration impact analysis with 10-minute threshold
- **Graphic 3**: Speaker count accuracy across systems  
- **Graphic 4**: Performance by accent type

### 3. Summary Statistics
- Average WER/cpWER per system
- Speaker count error rates
- Performance breakdown by accent and duration

## Troubleshooting

### Common Issues

1. **SpeechBrain Installation Failed**
```bash
# Try installing with specific torch version first
pip install torch==1.12.0 torchaudio==0.12.0
pip install speechbrain
```

2. **Accent Detection Model Download Issues**  
```bash
# Manually download model
from speechbrain.pretrained import EncoderClassifier
model = EncoderClassifier.from_hparams("Jzuluaga/accent-id-commonaccent_ecapa")
```

3. **Graphics Generation Errors**
```bash
# Install visualization dependencies
pip install matplotlib seaborn scipy
```

4. **Missing Audio Files**
- Ensure S3 credentials are correctly set
- Check that `audio_filepaths` dictionary contains valid file paths
- Verify audio file formats are supported (WAV, MP3, FLAC)

## Research Notes

### Why cpWER Instead of WSWER?
- **cpWER** is an established metric in the speech research community
- **Optimal Assignment**: Uses Hungarian algorithm for mathematically optimal speaker matching
- **Standardization**: Allows comparison with other research using the same metric


### Accent Detection Rationale
The `Jzuluaga/accent-id-commonaccent_ecapa` model was chosen because:
- **16 Accent Coverage**: Includes British, American, Southern US, and others
- **ECAPA-TDNN Architecture**: State-of-the-art for speaker and accent recognition
- **Research Validation**: Published and peer-reviewed approach
- **Hugging Face Integration**: Easy deployment and reproducibility

## Citation

If you use this code for research, please cite:
```bibtex
@misc{compare-diarization-wer-cpwer,
  title={Speaker Diarization Evaluation with WER and cpWER Metrics},
  author={Your Name},
  year={2024},
  url={https://github.com/mpathic/compare-diarization}
}
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-metric`)
3. Commit changes (`git commit -am 'Add new evaluation metric'`)
4. Push to branch (`git push origin feature/new-metric`)
5. Create Pull Request

## License

[Add your license information here]