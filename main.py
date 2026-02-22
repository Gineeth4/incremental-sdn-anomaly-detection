# main.py
import os, sys, subprocess
from src.utils.logger import get_logger

logger = get_logger()

def run_cmd_module(module_name):
    logger.info(f"Running module: {module_name}")
    cmd = [sys.executable, "-m", module_name]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error(f"Module {module_name} exited with code {result.returncode}")
        raise SystemExit(result.returncode)

def run_pipeline():
    logger.info("=== START PIPELINE ===")
    if os.path.exists("data/external") and any(f.lower().endswith(".csv") for f in os.listdir("data/external")):
        logger.info("Found CSV in data/external. Running adapter...")
        run_cmd_module("src.data_generation.real_data_adapter")
    else:
        logger.info("No CSV found in data/external/. Skipping adapter.")
    run_cmd_module("src.preprocessing.preprocess")
    run_cmd_module("src.models.base_model")
    run_cmd_module("src.models.incremental_model")
    run_cmd_module("src.visualization.visualize")
    run_cmd_module("src.evaluation.evaluate")
    logger.info("=== PIPELINE COMPLETE ===")

if __name__ == "__main__":
    run_pipeline()
