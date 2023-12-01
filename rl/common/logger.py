import logging
import os

class Logger():
    def __init__(self, log_path=None, log_level=None, screen_level=False):
        if log_level == None:
            self.log_level = logging.INFO
        else:
            self.log_level = log_level
        self.logger = logging.getLogger()
        self.logger.setLevel(self.log_level)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        if log_path:
            log_dir = "/".join(log_path.split("/")[:-1])
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.log_path = log_path
            fh = logging.FileHandler(self.log_path)
            fh.setLevel(self.log_level)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        if screen_level:
            ch = logging.StreamHandler()
            ch.setLevel(self.log_level)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def info(self, s):
        self.logger.info(s)

    def debug(self, s):
        self.logger.debug(s)

    def error(self, s):
        print("(ERROR!) {}".format(s))

log_level = logging.DEBUG
log_level = logging.INFO
#log_level = logging.ERROR
logger = Logger(None, log_level)
