import logging

logger = logging.getLogger('Logger')
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', filename='logs/log.log', level=logging.INFO)
