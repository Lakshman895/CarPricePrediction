import logging
import os
from datetime import datetime

main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
LOG_FILE = f"{datetime.now().strftime('%d_%m_%y_%H_%M_%S')}.log"
logs_path = os.path.join(main_dir,"logs")
os.makedirs(logs_path,exist_ok=True)
LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="[%(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"

)

if __name__ == "__main__":
    logging.info("Program started")