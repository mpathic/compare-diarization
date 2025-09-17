import os
import logging
import torch
import numpy as np
import librosa
from typing import Dict, List, Optional, Tuple
import warnings

logger = logging.getLogger(__name__)

class AccentDetector:
    """
    Accent detection module using Hugging Face models.
    Detects regional accents in English speech audio.
    """
    
    def __init__(self, model_name: str = "Jzuluaga/accent-id-commonaccent_ecapa", device: str = None):
        """
        Initialize the accent detector.
        
        Args:
            model_name: The Hugging Face model to use for accent detection
            device: Device to use ('cuda', 'cuda:0', 'mps', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name
        self.model = None
        
        # Device selection with auto-detection
        if device is None:
            if torch.cuda.is_available():
                self.device = f"cuda:{torch.cuda.current_device()}"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        logger.info(f"Accent detector will use device: {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load the accent detection model."""
        try:
            from speechbrain.pretrained import EncoderClassifier
            
            logger.info(f"Loading accent detection model: {self.model_name}")
            
            # Load the model with appropriate device settings
            run_opts = {"device": self.device}
            
            self.model = EncoderClassifier.from_hparams(
                source=self.model_name,
                savedir=f"pretrained_models/{self.model_name.split('/')[-1]}",
                run_opts=run_opts
            )
            
            logger.info(f"Accent detection model loaded successfully on {self.device}")
            
        except ImportError as e:
            logger.error("SpeechBrain not installed. Install with: pip install speechbrain")
            raise ImportError("SpeechBrain required for accent detection") from e
        except Exception as e:
            logger.error(f"Failed to load accent detection model: {e}")
            self.model = None
    
    def detect_accent(self, audio_path: str, top_k: int = 3) -> Dict:
        """
        Detect accent from audio file.
        
        Args:
            audio_path: Path to audio file
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary containing accent predictions and confidence scores
        """
        if self.model is None:
            logger.warning("Accent detection model not available")
            return {"error": "Model not loaded", "predictions": []}
        
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                return {"error": "Audio file not found", "predictions": []}
            
            logger.info(f"Detecting accent for: {audio_path}")
            
            # Perform accent classification
            out_prob, score, index, text_lab = self.model.classify_file(audio_path)
            
            # Convert outputs to numpy for easier handling
            if isinstance(out_prob, torch.Tensor):
                out_prob = out_prob.cpu().numpy()
            if isinstance(score, torch.Tensor):
                score = score.cpu().numpy()
            if isinstance(index, torch.Tensor):
                index = index.cpu().numpy()
            
            # Get the predicted accent - use raw model output
            predicted_accent = text_lab[0] if isinstance(text_lab, list) else str(text_lab)
            
            # Get top-k predictions
            if len(out_prob.shape) > 1:
                probabilities = out_prob[0]  # Take first sample if batch
            else:
                probabilities = out_prob
            
            # Get top k indices
            top_indices = np.argsort(probabilities)[-top_k:][::-1]
            
            predictions = []
            for idx in top_indices:
                # Use raw probability for each class
                predictions.append({
                    "accent": predicted_accent if idx == index else f"class_{idx}",
                    "confidence": float(probabilities[idx]),
                    "class_index": int(idx)
                })
            
            # Use raw model output as primary prediction
            primary_accent = predicted_accent
            
            result = {
                "primary_prediction": primary_accent,
                "primary_confidence": float(score) if np.isscalar(score) else float(score[0]),
                "top_predictions": predictions,
                "raw_output": {
                    "predicted_label": predicted_accent,
                    "confidence_score": float(score) if np.isscalar(score) else float(score[0]),
                    "class_index": int(index) if np.isscalar(index) else int(index[0])
                }
            }
            
            logger.info(f"Detected accent: {primary_accent} (confidence: {result['primary_confidence']:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during accent detection: {e}")
            return {"error": str(e), "predictions": []}
    
    def detect_accent_from_segments(self, audio_segments: List[np.ndarray], 
                                  sample_rate: int = 16000) -> Dict:
        """
        Detect accent from audio segments (useful for speaker-specific detection).
        
        Args:
            audio_segments: List of audio arrays
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary with accent predictions for each segment
        """
        if self.model is None:
            return {"error": "Model not loaded", "predictions": []}
        
        try:
            import tempfile
            import soundfile as sf
            
            results = []
            
            for i, segment in enumerate(audio_segments):
                if len(segment) < sample_rate * 0.5:  # Skip very short segments
                    logger.debug(f"Skipping short segment {i} (duration: {len(segment)/sample_rate:.2f}s)")
                    continue
                
                # Save segment to temporary file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    sf.write(tmp_file.name, segment, sample_rate)
                    
                    # Detect accent for this segment
                    segment_result = self.detect_accent(tmp_file.name)
                    segment_result['segment_id'] = i
                    segment_result['segment_duration'] = len(segment) / sample_rate
                    results.append(segment_result)
                    
                    # Clean up temporary file
                    os.unlink(tmp_file.name)
            
            # Aggregate results (could use voting, confidence weighting, etc.)
            if results:
                # Simple approach: use highest confidence prediction
                best_result = max(results, key=lambda x: x.get('primary_confidence', 0))
                
                return {
                    "aggregated_prediction": best_result['primary_prediction'],
                    "aggregated_confidence": best_result['primary_confidence'],
                    "segment_results": results,
                    "total_segments_processed": len(results)
                }
            else:
                return {"error": "No valid segments for processing", "predictions": []}
                
        except Exception as e:
            logger.error(f"Error in segment-based accent detection: {e}")
            return {"error": str(e), "predictions": []}
    
    def batch_detect_accents(self, audio_paths: List[str]) -> Dict[str, Dict]:
        """
        Detect accents for multiple audio files.
        
        Args:
            audio_paths: List of paths to audio files
            
        Returns:
            Dictionary mapping file paths to accent predictions
        """
        results = {}
        
        for audio_path in audio_paths:
            logger.info(f"Processing {audio_path} for accent detection...")
            results[audio_path] = self.detect_accent(audio_path)
        
        return results
    
    def get_supported_accents(self) -> List[str]:
        """
        Get list of accent categories that the model can detect.
        Note: This returns the raw model output labels, not mapped names.
        
        Returns:
            List of model's accent class labels
        """
        # Since we're using raw output, we can't predetermine the exact labels
        # The model will return labels like 'us', 'england', 'australia', etc.
        return ["Raw model outputs - see model documentation for full list"]
    
    def classify_accent_region(self, detected_accent: str) -> str:
        """
        Classify detected accent into broader regional categories.
        
        Args:
            detected_accent: The raw detected accent label from model
            
        Returns:
            Broader regional category (basic geographic grouping)
        """
        # Basic geographic grouping based on common model outputs
        accent_lower = detected_accent.lower()
        
        if any(region in accent_lower for region in ['us', 'america', 'canada']):
            return 'North American'
        elif any(region in accent_lower for region in ['england', 'british', 'ireland', 'scotland', 'wales']):
            return 'British Isles'
        elif any(region in accent_lower for region in ['australia', 'newzealand']):
            return 'Oceanic'
        elif any(region in accent_lower for region in ['india', 'singapore', 'malaysia', 'philippines']):
            return 'Asian English'
        elif any(region in accent_lower for region in ['africa', 'south']):
            return 'African/Southern'
        else:
            return 'Other'


# Example usage and testing functions
def test_accent_detector():
    """Test function for accent detector."""
    detector = AccentDetector()
    
    # Test with a sample file (you would replace with actual test files)
    test_files = [
        "path/to/american_sample.wav",
        "path/to/british_sample.wav", 
        "path/to/southern_sample.wav"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            result = detector.detect_accent(test_file)
            print(f"File: {test_file}")
            print(f"Detected accent: {result}")
            print("-" * 50)


if __name__ == "__main__":
    # Set up logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Test the accent detector
    test_accent_detector()