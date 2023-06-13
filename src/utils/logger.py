import logging
import os

def get_logger(log_dir):
    # 디렉토리가 없으면 생성
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Initialize logger
    logger = logging.getLogger()

    # Create handler for writing to log file
    file_handler = logging.FileHandler(os.path.join(log_dir, 'train.log'))

    # Create handler for printing to console
    stream_handler = logging.StreamHandler()

    # Set level of logging
    logger.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    stream_handler.setLevel(logging.INFO)

    # Format the logs
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

