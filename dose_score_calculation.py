import os
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import mean_absolute_error
from config import PATH_CONFIG

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_csv(file_path):
    """Load dose data from a CSV file."""
    logger.debug(f"Loading CSV file from {file_path}")
    return pd.read_csv(file_path).values