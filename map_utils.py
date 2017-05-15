# -*- coding: utf-8 -*-
from PIL import Image, ImageChops


class TileProvider():
    def __init__(self, input_file_name, square_size):
        self.map = Image.open(input_file_name)
        self.square_size = square_size


    def load_tile(x, y):
        tile_box = (y, x, y + self.square_size, x + self.square_size)
        tile = self.map.crop(tile_box)
        
        data = np.asarray(tile, dtype="int32")
        return data


class Visualizer():
    def __init__(self, square_size, color_map):
        self.square_size = square_size
        self.covers = [Image.new("RGB", (square_size, square_size), color) for color in color_map]


    def colorize(self, image, color):
        cover = self.covers[color]
        colorized = ImageChops.add(image, cover, 2.0)
        return colorized


    def run(self, input_file_name, output_file_name, tile_labels):
        original_image = Image.open(input_file_name)
        image_width, image_height = original_image.size

        colorized_image = Image.new("RGB", original_image.size)

        tile_index = 0
        for i in range(0, image_height, self.square_size):
            for j in range(0, image_width, self.square_size):
                tile_box = (j, i, j + self.square_size, i + self.square_size)
                label = tile_labels[tile_index]

                tile = original_image.crop(tile_box)
                colorized_tile = self.colorize(tile, label)
                colorized_image.paste(colorized_tile, tile_box)

                #print("Tile %d has colorized" % (tile_index,))
                tile_index += 1

        colorized_image.save(output_file_name)