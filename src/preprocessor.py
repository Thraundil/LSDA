def preprocess(img):
    """
    The image preprocessor. Receives a PIL image and returns one.
    Called by the scraper in a multithreaded enviroment.
    """
    return img