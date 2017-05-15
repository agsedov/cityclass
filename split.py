# -*- coding: utf-8 -*-

import config as cnf
from PIL import Image

# Скриптик для того чтобы разбить картинку на квадратики
def split_to_tiles(path_to_save, input_file_name, square_size):
    image = Image.open(input_file_name)
    image_width, image_height = image.size

    for i in range(0, image_height, square_size):
        for j in range(0, image_width, square_size):
            box = (j, i, j + square_size, i + square_size)
            tile = image.crop(box)
            
            filename = path_to_save + "/tile-%d-%d.png" % (i // square_size, j // square_size)
            tile.save(filename)
            print(filename + ' created')
            

split_to_tiles(cnf.tiles, cnf.map_source, cnf.square_size)