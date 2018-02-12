import logging

#class SystemLog:
#    def __init__(self, file, level1, level2):
#        logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
#        rootLogger = logging.getLogger()
#        fileHandler = logging.FileHandler(file)
#        fileHandler.setFormatter(logFormatter)
#        fileHandler.setLevel(level1)
#        rootLogger.addHandler(fileHandler)
#        consoleHandler = logging.StreamHandler()
#        consoleHandler.setFormatter(logFormatter)
#        rootLogger.addHandler(consoleHandler)
#        rootLogger.setLevel(level2)
#        self.log = logging

import os
import time
import datetime
import logging

from horus.core.config import HorusConfig


class SysLogger:
    def __init__(self):
        self.logger = logging.getLogger('horus')
        self.conf = HorusConfig()
        if len(self.logger.handlers) == 0:
            self.logger.setLevel(logging.DEBUG)
            now = datetime.datetime.now()
            handler = logging.FileHandler(self.conf.root_dir + 'log/horus_'
                                          + now.strftime("%Y-%m-%d")
                                          + '.log')
            formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            consoleHandler = logging.StreamHandler()
            consoleHandler.setFormatter(formatter)
            self.logger.addHandler(consoleHandler)
    def getLog(self):
        return self.logger