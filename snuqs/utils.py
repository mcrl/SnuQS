import logging

logger = logging.getLogger('SnuQS')
sh = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s][%(filename)s:%(lineno)d:%(funcName)s] %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)
logger.setLevel(logging.INFO)
