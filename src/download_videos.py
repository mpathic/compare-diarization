import os
import re
import csv
import sys
import logging
import time

import yt_dlp

logger = logging.getLogger(__name__)  # respect mains loglevel

def load_urls_from_file(file_path):
    """Load URLs from a CSV or text file"""
    logger.info(f"Loading URLs from file: {file_path}")
    
    if file_path.endswith('.csv'):
        try:
            df = pd.read_csv(file_path)
            logger.debug(f"CSV loaded with columns: {df.columns.tolist()}")
            
            url_column = next((col for col in ['url', 'video_url', 'youtube_url'] if col in df.columns), None)
            if url_column:
                urls = df[url_column].tolist()
                logger.info(f"Found {len(urls)} URLs in column '{url_column}'")
                return urls
            else:
                error_msg = f"Could not find URL column in {file_path}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            raise
    else:
        # Assume TXT file has one URL per line
        try:
            with open(file_path, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(urls)} URLs from text file")
            return urls
        except Exception as e:
            logger.error(f"Error loading TXT file: {str(e)}")
            raise


def extract_video_id(url):
    """Extract the video ID from a YouTube URL"""
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    return None


def sanitize_filename(title):
    sanitized = re.sub(r'[^\w\-]', '_', title)
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip('_')
    return sanitized


def download_audio(url, output_dir, file_prefix):
    """Download audio from a YouTube URL using yt-dlp"""
    output_template = os.path.join(output_dir, f"{file_prefix}_%(title)s.%(ext)s")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
        # Add retries for transient network issues
        'retries': 5,
        'fragment_retries': 5,
        'skip_unavailable_fragments': True,
    }
    
    logger.debug(f"Starting download using yt-dlp from {url}")
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if not info:
                logger.error(f"Failed to extract info from URL: {url}")
                return None, None
                
            downloaded_file = ydl.prepare_filename(info)
            
            # Get the base filename without extension
            base_filename = downloaded_file.rsplit('.', 1)[0]
            
            # Sanitize the filename - replace spaces with underscores
            title = info.get('title', 'Unknown')
            sanitized_title = sanitize_filename(title)
            
            # Create new filename with sanitized title
            new_base_filename = os.path.join(output_dir, f"{file_prefix}_{sanitized_title}")
            mp3_filename = f"{new_base_filename}.mp3"
            
            # Rename the file if it exists
            old_mp3_filename = f"{base_filename}.mp3"
            if os.path.exists(old_mp3_filename):
                if old_mp3_filename != mp3_filename:
                    try:
                        os.rename(old_mp3_filename, mp3_filename)
                        logger.debug(f"Renamed file to: {mp3_filename}")
                    except OSError as e:
                        logger.warning(f"Could not rename file {old_mp3_filename} to {mp3_filename}: {str(e)}")
                        mp3_filename = old_mp3_filename  # Use the original filename
                return mp3_filename, info
            else:
                logger.error(f"Expected mp3 file not found at {old_mp3_filename}")
                return None, info  # Return None to indicate failure
                    
    except yt_dlp.utils.DownloadError as e:
        logger.error(f"yt-dlp download error for {url}: {str(e)}")
        return None, None
    except Exception as e:
        logger.error(f"Unexpected error downloading audio from {url}: {str(e)}", exc_info=True)
        return None, None


def download(url, transcript_id=None, output_dir="downloads", max_retries=3):
    """
    Download YouTube videos from the ANNO MI dataset
    """
    logger.info(f"Starting download process of {url} to {output_dir} ...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    total_bytes_downloaded = 0
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Extract video ID from the URL (only for fallback)
            video_id = extract_video_id(url)
            if not video_id:
                logger.warning(f"Could not extract video ID from URL: {url}, will use a placeholder")
                video_id = "unknown_video"

            # Use transcript_id for filename if provided, otherwise use video_id
            file_prefix = transcript_id if transcript_id else video_id
            mp3_filename, info = download_audio(url, output_dir, file_prefix)

            if mp3_filename is None:
                logger.warning(f"Download failed for URL: {url}. Attempt {retry_count + 1}/{max_retries}")
                retry_count += 1
                time.sleep(2)  # Wait before retrying
                continue

            # Check if the file exists and has content
            if os.path.exists(mp3_filename):
                file_size = os.path.getsize(mp3_filename)
                if file_size > 0:
                    total_bytes_downloaded += file_size
                    logger.info(f"Downloaded: {os.path.basename(mp3_filename)} ({file_size/1024/1024:.2f} MB)")
                    return mp3_filename
                else:
                    logger.warning(f"Downloaded file has zero size: {mp3_filename}")
                    os.remove(mp3_filename)  # Remove empty file
                    retry_count += 1
                    continue
            else:
                logger.warning(f"Expected file not found after download: {mp3_filename}")
                retry_count += 1
                continue

        except Exception as e:
            logger.error(f"Error downloading {url}: {str(e)}", exc_info=True)
            retry_count += 1
            if retry_count < max_retries:
                logger.info(f"Retrying download ({retry_count}/{max_retries})...")
                time.sleep(2)  # Wait before retrying
            continue

    # If we've exhausted all retries
    error_msg = f"Failed to download audio after {max_retries} attempts from URL: {url} - skipping this video"
    logger.error(error_msg)
    
    # Return None to indicate failure
    return None


def main():
    logger.info("YT Script execution")


if __name__ == "__main__":
    main()