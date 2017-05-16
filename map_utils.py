# -*- coding: utf-8 -*-

from numpy import asarray
from PIL import Image, ImageChops

class TileDataProvider():
    def __init__(self, input_file_name, square_size):
        self.map = Image.open(input_file_name)
        self.square_size = square_size


    def load_tile_data(self, x, y):
        tile_box = (y * self.square_size, x * self.square_size, 
                    (y + 1) * self.square_size, (x + 1) * self.square_size)

        tile = self.map.crop(tile_box)
        
        data = asarray(tile, dtype="int32")
        return data

    def destruct(self):
        self.map.close()
