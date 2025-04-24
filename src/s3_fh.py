import os
import logging
import boto3
import re

logger = logging.getLogger(__name__)  # respect main's loglevel

def download_from_s3(bucket_name, s3_key, download_path):
    """
    Download a file from an S3 bucket.
    :param bucket_name: Name of the S3 bucket.
    :param s3_key: Path to the file in the S3 bucket.
    :param download_path: Local path to save the downloaded file.
    """
    s3 = boto3.client('s3')
    try:
        # Download the file
        s3.download_file(bucket_name, s3_key, download_path)
        logger.info(f"File downloaded successfully to {download_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return False

def s3_download_files():
    """
    Download up to 100 files from the S3 bucket and return a mapping of transcript_id to file path.
    Returns a dictionary of transcript_id to local_filepath
    """
    # Create directory to store the downloaded files
    os.makedirs('audio_files', exist_ok=True)
    
    S3_BUCKET_NAME = 'diarization-audiofiles'
    s3 = boto3.client('s3')
    
    # Dictionary to store transcript_id to local file path mapping
    transcript_map = {}
    
    try:
        # List objects in the bucket (up to 200)
        response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME)
        
        if 'Contents' not in response:
            logger.warning(f"No files found in bucket {S3_BUCKET_NAME}")
            return transcript_map
            
        logger.info(f"Found {len(response['Contents'])} files in bucket")
        
        # Process each file
        for obj in response['Contents']:
            s3_key = obj['Key']
            filename = os.path.basename(s3_key)
            
            # Extract transcript_id from filename (assuming format "{int}_textetc")
            match = re.match(r'(\d+)_.*', filename)
            if match:
                transcript_id = match.group(1)
                local_path = os.path.join('audio_files', filename)
                
                # Download the file
                logger.info(f"Downloading {s3_key} to {local_path}...")
                if download_from_s3(S3_BUCKET_NAME, s3_key, local_path):
                    transcript_map[transcript_id] = local_path
            else:
                logger.warning(f"Couldn't extract transcript_id from filename: {filename}")
                
    except Exception as e:
        logger.error(f"Error processing S3 bucket: {e}")
    
    logger.info(f"Downloaded {len(transcript_map)} files with valid transcript IDs")
    return transcript_map

def main():
    logger.info("Starting S3 Pull ...")
    transcript_mapping = download_files()
    logger.info(f"Completed with {len(transcript_mapping)} files mapped")
    return transcript_mapping

if __name__ == '__main__':
    main()