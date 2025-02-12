import os
import numpy as np
import pandas as pd
import logging
import datetime
from sklearn.metrics import mean_absolute_error
from config import PATH_CONFIG

# Configure metadata
CURRENT_TIME = "2025-02-02 10:16:41"
CURRENT_USER = "aagneye-syam"

# Configure logging with timestamp
log_filename = f'ensemble_log_{CURRENT_TIME.replace(" ", "_").replace(":", "-")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename)
    ]
)
logger = logging.getLogger(__name__)