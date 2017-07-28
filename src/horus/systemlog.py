import logging


class SystemLog:

    def __init__(self, file, level1, level2):
        logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        rootLogger = logging.getLogger()
        fileHandler = logging.FileHandler(file)
        fileHandler.setFormatter(logFormatter)
        fileHandler.setLevel(level1)
        rootLogger.addHandler(fileHandler)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)
        logging.getLogger("").setLevel(level2)
        self.log = logging
