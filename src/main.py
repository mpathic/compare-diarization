import argparse
import logging
import os
import json
import sys
import time

from src.s3_fh import download_file
from src.processors import td_revai, td_assemblyai, td_opensource
# from src.comparison.metrics import calculate_all_metrics

## USAGE (from root directory of project):
## python3 -m src.main


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def main():

    logger.info("starting ...")
    logger.info("Workflow finished.")


if __name__ == "__main__":
    main()