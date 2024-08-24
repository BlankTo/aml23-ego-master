import logging
import coloredlogs
from utils.args import args
import sys
import os
import shutil


def setup_logger(name, logfile=None):

    logger_instance = logging.getLogger(name)
    i_handler = logging.FileHandler(logfile)
    i_handler.setLevel(logging.INFO)
    logger_instance.addHandler(i_handler)
    coloredlogs.install(
        level='DEBUG', logger=logger_instance,
        fmt='%(asctime)s %(name)s %(levelname)s %(message)s')
    return logger_instance


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

def clean_false_log():
    for folder in os.listdir(os.path.dirname(os.path.dirname(args.logfile))):
        folder_path = os.path.join(os.path.dirname(os.path.dirname(args.logfile)), folder)
        for file in os.listdir(folder_path):
            if os.path.getsize(os.path.join(folder_path, file)) == 0:
                try:
                    shutil.rmtree(folder_path)
                except: pass


sys.excepthook = handle_exception
logger = setup_logger("LOG", args.logfile)
