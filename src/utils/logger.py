# src/utils/logger.py
from loguru import logger
import sys
import os

LOG_DIR = "experiments/results/logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}")
logger.add(f"{LOG_DIR}/system.log", level="DEBUG", rotation="5 MB", retention="7 days")
def get_logger():
    return logger
    