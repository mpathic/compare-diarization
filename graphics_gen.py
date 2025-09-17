import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from typing import Dict, List, Tuple
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PosterGraphicsGenerator:
    """Generate graphics for ISTCM poster presentation."""
    
    def __init__(self, results_file: str = 'out/evaluation_results_with_wer_cpwer.json'):
        """
        Initialize with results file.
        
        Args:
            results_file: Path to the evaluation results JSON file
        """
        self.results_file = results_file
        self.results_data = {}
        self.summary_data = {}
        self.load_data()
        
        # Create output directory for graphics
        os.makedirs('out/graphics', exist_ok=True)
        
    def load_data(self):
        """Load results data from JSON file."""
        try:
            with open(self.results_file, 'r') as f:
                self.results_data = json.load(f)
            logger.info(f"Loaded {len(self.results_data)} transcript results")
            
            # Try to load summary data
            summary_file = 'out/summary_statistics.json'
            if os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    self.summary_data = json.load(f)
                logger.info("Loaded summary statistics")
            else:
                logger.warning("Summary statistics file not found, will generate from raw data")
                self._generate_summary_data()
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _generate_summary_data(self):
        """Generate summary data if not available."""
        logger.info("Generating summary data from results...")
        
        processors = ['revai', 'assemblyai', 'opensource']
        
        # Initialize summary structure
        self.summary_data = {
            'overall_performance': {},
            'speaker_count_accuracy': {},
            'duration_analysis': [],
            'accent_analysis': {}
        }
        
        # Collect data for each processor
        for processor in processors:
            wers = []
            cpwers = []
            speaker_errors = 0
            total_files = 0
            
            for transcript_id, data in self.results_data.items():
                if processor in data:
                    total_files += 1
                    
                    # WER data
                    wer_metrics = data[processor].get('wer_metrics', {})
                    if 'wer' in wer_metrics and wer_metrics['wer'] != float('inf'):
                        wers.append(wer_metrics['wer'])
                    
                    # cpWER data
                    cpwer_metrics = data[processor].get('cpwer_metrics', {})
                    if 'cpwer' in cpwer_metrics and cpwer_metrics['cpwer'] != float('inf'):
                        cpwers.append(cpwer_metrics['cpwer'])
                        
                        # Duration analysis
                        duration_min = data.get('audio_duration_minutes', 0)
                        self.summary_data['duration_analysis'].append({
                            'duration_min': duration_min,
                            'cpwer': cpwer_metrics['cpwer'],
                            'processor': processor,
                            'transcript_id': transcript_id
                        })
                    
                    # Speaker count errors
                    speaker_metrics = data[processor].get('speaker_count_metrics', {})
                    if not speaker_metrics.get('speaker_count_correct', True):
                        speaker_errors += 1
            
            # Store summary statistics
            self.summary_data['overall_performance'][processor] = {
                'avg_wer': float(np.mean(wers)) if wers else float('inf'),
                'avg_cpwer': float(np.mean(cpwers)) if cpwers else float('inf'),
                'wer_std': float(np.std(wers)) if wers else 0,
                'cpwer_std': float(np.std(cpwers)) if cpwers else 0,
                'num_files': len(wers)
            }
            
            self.summary_data['speaker_count_accuracy'][processor] = {
                'total_files': total_files,
                'incorrect_speaker_count': speaker_errors,
                'error_percentage': (speaker_errors / total_files * 100) if total_files > 0 else 0
            }
    
    def generate_graphic_1_headline_results(self):
        """
        Graphic 1: Headline Results - Overall Performance
        Grouped bar chart showing WER vs cpWER for each system.
        """
        logger.info("Generating Graphic 1: Headline Results")
        
        # Extract data
        processors = ['AssemblyAI', 'RevAI', 'Whisper + Pyannote']
        processor_keys = ['assemblyai', 'revai', 'opensource']
        
        wer_values = []
        cpwer_values = []
        
        for key in processor_keys:
            perf = self.summary_data['overall_performance'][key]
            wer_values.append(perf['avg_wer'] * 100)  # Convert to percentage
            cpwer_values.append(perf['avg_cpwer'] * 100)  # Convert to percentage
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(processors))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, wer_values, width, label='Overall WER (%)', 
                      color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, cpwer_values, width, label='Diarized cpWER (%)', 
                      color='#A23B72', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Customize the plot
        ax.set_xlabel('System', fontsize=14, fontweight='bold')
        ax.set_ylabel('Error Rate (%)', fontsize=14, fontweight='bold')
        ax.set_title('Overall Performance: WER vs cpWER\nThe Trade-off Between Transcription and Speaker Attribution Accuracy', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(processors, fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        
        # Set y-axis limit with some headroom
        max_val = max(max(wer_values), max(cpwer_values))
        ax.set_ylim(0, max_val * 1.15)
        
        plt.tight_layout()
        plt.savefig('out/graphics/graphic_1_headline_results.png', dpi=300, bbox_inches='tight')
        plt.savefig('out/graphics/graphic_1_headline_results.pdf', bbox_inches='tight')
        plt.show()
        
        logger.info("Graphic 1 saved as PNG and PDF")
    
    def generate_graphic_2_duration_cliff(self):
        """
        Graphic 2: The "10-Minute Cliff" - Impact of Audio Duration
        Scatter plot with smoothed trendlines showing cpWER vs duration.
        """
        logger.info("Generating Graphic 2: Duration Cliff Analysis")
        
        # Prepare data
        duration_data = pd.DataFrame(self.summary_data['duration_analysis'])
        
        if duration_data.empty:
            logger.warning("No duration data available for plotting")
            return
        
        # Convert cpWER to percentage
        duration_data['cpwer_pct'] = duration_data['cpwer'] * 100
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Define colors for each system
        colors = {'revai': '#E63946', 'assemblyai': '#2E86AB', 'opensource': '#06D6A0'}
        labels = {'revai': 'RevAI', 'assemblyai': 'AssemblyAI', 'opensource': 'Whisper + Pyannote'}
        
        # Plot data points and trendlines for each processor
        for processor in ['revai', 'assemblyai', 'opensource']:
            proc_data = duration_data[duration_data['processor'] == processor]
            
            if proc_data.empty:
                continue
                
            x_vals = proc_data['duration_min'].values
            y_vals = proc_data['cpwer_pct'].values
            
            # Scatter plot
            ax.scatter(x_vals, y_vals, color=colors[processor], alpha=0.7, s=50, 
                      label=labels[processor], edgecolor='black', linewidth=0.5)
            
            # Add trendline using polynomial fit and Gaussian smoothing
            if len(x_vals) > 3:
                # Sort data for proper line plotting
                sorted_indices = np.argsort(x_vals)
                x_sorted = x_vals[sorted_indices]
                y_sorted = y_vals[sorted_indices]
                
                # Create smooth x values for trendline
                x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 100)
                
                # Polynomial fit (degree 2)
                coeffs = np.polyfit(x_sorted, y_sorted, 2)
                y_trend = np.polyval(coeffs, x_smooth)
                
                # Apply Gaussian smoothing
                y_smooth = gaussian_filter1d(y_trend, sigma=2)
                
                ax.plot(x_smooth, y_smooth, color=colors[processor], linewidth=2, alpha=0.8)
        
        # Add the "10-minute cliff" vertical line
        ax.axvline(x=10, color='red', linestyle='--', linewidth=2, alpha=0.8)
        ax.text(10.2, ax.get_ylim()[1] * 0.9, 'Performance\nThreshold\n(10 min)', 
               fontsize=12, fontweight='bold', color='red')
        
        # Customize the plot
        ax.set_xlabel('Audio Duration (Minutes)', fontsize=14, fontweight='bold')
        ax.set_ylabel('cpWER (%)', fontsize=14, fontweight='bold')
        ax.set_title('The "10-Minute Cliff": Impact of Audio Duration on Speaker Attribution Accuracy', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Set reasonable axis limits
        ax.set_xlim(0, max(duration_data['duration_min']) * 1.05)
        ax.set_ylim(0, max(duration_data['cpwer_pct']) * 1.1)
        
        plt.tight_layout()
        plt.savefig('out/graphics/graphic_2_duration_cliff.png', dpi=300, bbox_inches='tight')
        plt.savefig('out/graphics/graphic_2_duration_cliff.pdf', bbox_inches='tight')
        plt.show()
        
        logger.info("Graphic 2 saved as PNG and PDF")
    
    def generate_graphic_3_speaker_count_accuracy(self):
        """
        Graphic 3: Speaker Count Accuracy
        Bar chart showing percentage of files with incorrect speaker count.
        """
        logger.info("Generating Graphic 3: Speaker Count Accuracy")
        
        # Extract data
        processors = ['AssemblyAI', 'RevAI', 'Whisper + Pyannote']
        processor_keys = ['assemblyai', 'revai', 'opensource']
        
        error_percentages = []
        for key in processor_keys:
            accuracy_data = self.summary_data['speaker_count_accuracy'][key]
            error_percentages.append(accuracy_data['error_percentage'])
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Use different colors to highlight the problematic system
        colors = ['#06D6A0' if x < 20 else '#E63946' for x in error_percentages]
        
        bars = ax.bar(processors, error_percentages, color=colors, alpha=0.8, 
                     edgecolor='black', linewidth=0.5)
        
        # Customize the plot
        ax.set_xlabel('System', fontsize=14, fontweight='bold')
        ax.set_ylabel('Files with Incorrect Speaker Count (%)', fontsize=14, fontweight='bold')
        ax.set_title('Speaker Count Accuracy\nA Critical Flaw in Pure Transcription-Based Evaluation', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Set y-axis limit with headroom
        ax.set_ylim(0, max(error_percentages) * 1.15)
        
        plt.tight_layout()
        plt.savefig('out/graphics/graphic_3_speaker_count_accuracy.png', dpi=300, bbox_inches='tight')
        plt.savefig('out/graphics/graphic_3_speaker_count_accuracy.pdf', bbox_inches='tight')
        plt.show()
        
        logger.info("Graphic 3 saved as PNG and PDF")
    
    def generate_graphic_4_accent_robustness(self):
        """
        Graphic 4: Accent Robustness
        Grouped bar chart showing cpWER for different accents.
        """
        logger.info("Generating Graphic 4: Accent Robustness")
        
        # Collect accent performance data from raw results
        accent_performance = {}
        
        for transcript_id, data in self.results_data.items():
            accent = data.get('accent', 'American')  # Use detected accent, default to American
            
            if accent not in accent_performance:
                accent_performance[accent] = {'revai': [], 'assemblyai': [], 'opensource': []}
            
            for processor in ['revai', 'assemblyai', 'opensource']:
                if processor in data:
                    cpwer_metrics = data[processor].get('cpwer_metrics', {})
                    if 'cpwer' in cpwer_metrics and cpwer_metrics['cpwer'] != float('inf'):
                        accent_performance[accent][processor].append(cpwer_metrics['cpwer'] * 100)
        
        # Filter out accents with insufficient data
        filtered_accents = {}
        for accent, data in accent_performance.items():
            if any(len(data[proc]) > 0 for proc in ['revai', 'assemblyai', 'opensource']):
                filtered_accents[accent] = data
        
        if not filtered_accents:
            logger.warning("No accent data available for plotting")
            return
        
        # Prepare data for plotting
        accents = list(filtered_accents.keys())
        processors = ['RevAI', 'AssemblyAI', 'Whisper + Pyannote']
        processor_keys = ['revai', 'assemblyai', 'opensource']
        
        # Calculate means for each accent-processor combination
        plot_data = np.zeros((len(accents), len(processors)))
        
        for i, accent in enumerate(accents):
            for j, proc_key in enumerate(processor_keys):
                values = filtered_accents[accent][proc_key]
                if values:
                    plot_data[i, j] = np.mean(values)
                else:
                    plot_data[i, j] = np.nan
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(accents))
        width = 0.25
        colors = ['#E63946', '#2E86AB', '#06D6A0']
        
        bars = []
        for i, (processor, color) in enumerate(zip(processors, colors)):
            values = plot_data[:, i]
            # Replace NaN with 0 for plotting (will be handled in labels)
            values_plot = np.where(np.isnan(values), 0, values)
            bar = ax.bar(x + (i - 1) * width, values_plot, width, 
                        label=processor, color=color, alpha=0.8, 
                        edgecolor='black', linewidth=0.5)
            bars.append(bar)
        
        # Customize the plot
        ax.set_xlabel('Accent Type', fontsize=14, fontweight='bold')
        ax.set_ylabel('cpWER (%)', fontsize=14, fontweight='bold')
        ax.set_title('Accent Robustness: Performance Varies by Regional Accent\nHighlighting Model Bias in Speech Recognition', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(accents, fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars (skip NaN values)
        for i, bar_group in enumerate(bars):
            for j, bar in enumerate(bar_group):
                height = bar.get_height()
                if height > 0 and not np.isnan(plot_data[j, i]):  # Only label non-zero, non-NaN values
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{height:.1f}%', ha='center', va='bottom', 
                           fontweight='bold', fontsize=10)
        
        # Set y-axis limit
        max_val = np.nanmax(plot_data)
        if not np.isnan(max_val):
            ax.set_ylim(0, max_val * 1.15)
        
        plt.tight_layout()
        plt.savefig('out/graphics/graphic_4_accent_robustness.png', dpi=300, bbox_inches='tight')
        plt.savefig('out/graphics/graphic_4_accent_robustness.pdf', bbox_inches='tight')
        plt.show()
        
        logger.info("Graphic 4 saved as PNG and PDF")
    
    def generate_all_graphics(self):
        """Generate all four poster graphics."""
        logger.info("Generating all poster graphics...")
        
        try:
            self.generate_graphic_1_headline_results()
            self.generate_graphic_2_duration_cliff()
            self.generate_graphic_3_speaker_count_accuracy()
            self.generate_graphic_4_accent_robustness()
            
            logger.info("All graphics generated successfully!")
            logger.info("Graphics saved in 'out/graphics/' directory")
            
        except Exception as e:
            logger.error(f"Error generating graphics: {e}")
            raise
    
    def generate_summary_table(self):
        """Generate a summary table of key metrics."""
        logger.info("Generating summary table...")
        
        # Create summary DataFrame
        summary_rows = []
        
        for processor_key in ['assemblyai', 'revai', 'opensource']:
            processor_name = {'assemblyai': 'AssemblyAI', 
                            'revai': 'RevAI', 
                            'opensource': 'Whisper + Pyannote'}[processor_key]
            
            perf = self.summary_data['overall_performance'][processor_key]
            spk = self.summary_data['speaker_count_accuracy'][processor_key]
            
            summary_rows.append({
                'System': processor_name,
                'Average WER (%)': f"{perf['avg_wer'] * 100:.2f}",
                'Average cpWER (%)': f"{perf['avg_cpwer'] * 100:.2f}",
                'Speaker Count Errors (%)': f"{spk['error_percentage']:.1f}",
                'Files Processed': perf['num_files']
            })
        
        df = pd.DataFrame(summary_rows)
        
        # Save as CSV
        df.to_csv('out/graphics/summary_table.csv', index=False)
        
        # Create a nice table visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        
        # Header styling
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(df) + 1):
            for j in range(len(df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.title('Summary of Key Performance Metrics', fontsize=16, fontweight='bold', pad=20)
        plt.savefig('out/graphics/summary_table.png', dpi=300, bbox_inches='tight')
        plt.savefig('out/graphics/summary_table.pdf', bbox_inches='tight')
        plt.show()
        
        logger.info("Summary table saved")
        print("\nSummary Table:")
        print(df.to_string(index=False))


def main():
    """Main function to generate all graphics."""
    try:
        # Check if results file exists
        results_file = 'out/evaluation_results_with_wer_cpwer.json'
        if not os.path.exists(results_file):
            logger.error(f"Results file not found: {results_file}")
            logger.error("Please run the main evaluation script first to generate results.")
            return
        
        # Generate graphics
        generator = PosterGraphicsGenerator(results_file)
        generator.generate_all_graphics()
        generator.generate_summary_table()
        
        print("\n" + "="*60)
        print("POSTER GRAPHICS GENERATION COMPLETE")
        print("="*60)
        print(f"Graphics saved in: out/graphics/")
        print("\nGenerated files:")
        print("- graphic_1_headline_results.png/pdf")
        print("- graphic_2_duration_cliff.png/pdf") 
        print("- graphic_3_speaker_count_accuracy.png/pdf")
        print("- graphic_4_accent_robustness.png/pdf")
        print("- summary_table.png/pdf/csv")
        print("\nAll graphics are publication-ready at 300 DPI.")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()