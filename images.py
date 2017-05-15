# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image

import config as cnf

def load_image(filename) :
    img = Image.open(filename)
    img.load()

    data = np.asarray(img, dtype="int32")
    return data

def get_filename_for_tile(x, y):
	print(cnf.tiles + "/tile-%d-%d.png" % (x, y))
	return cnf.tiles + "/tile-%d-%d.png" % (x, y)

def load_tile(x, y):
	return load_image(get_filename_for_tile(x, y))
