import logging
from datetime import datetime

# Set up logging
current_time = "2025-02-07 05:37:38"
current_user = "aagneye-syam"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - User: ' + current_user + ' - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

