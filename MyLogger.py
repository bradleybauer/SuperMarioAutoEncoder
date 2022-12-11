import logging

# from https://github.com/yccyenchicheng/pytorch-WGAN-GP-TTUR-CelebA
def get_logger(name, logpath, filepath='', package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger(name)
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        try:
            info_file_handler = logging.FileHandler(logpath, mode="a")
        except FileNotFoundError:
            info_file_handler = logging.FileHandler(logpath, mode="w")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    if filepath:
        logger.info(filepath)
        with open(filepath, "r") as f:
            logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger

