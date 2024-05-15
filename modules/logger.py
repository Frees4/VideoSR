import logging
import os
from modules.shared import LOG_DIR, LOG_FILE

logger = logging.getLogger('Logger')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', filename=LOG_DIR+LOG_FILE, level=logging.INFO)
