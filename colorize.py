# -*- coding: utf-8 -*-

from numpy import asarray, loadtxt
from PIL import Image, ImageChops

import config as cfg


def create_covering(tile_labels, color_map):  
    covers = { label: Image.new("RGB", (cfg.square_size, cfg.square_size), color) for (label, color) in color_map.items() } 

    rows, columns = tile_labels.shape
    covering = Image.new("RGB", (cfg.square_size * columns, cfg.square_size * rows))

    for row in range(rows):
        for column in range(columns):
            label = tile_labels[row, column]
            cover = covers[label]

            tile_box = (column * cfg.square_size, row * cfg.square_size, (column + 1) * cfg.square_size, (row + 1) * cfg.square_size)
            covering.paste(cover, tile_box)

    return covering


tile_labels = loadtxt(cfg.labels_source, delimiter=' ').astype('int'); 
citymap = Image.open(cfg.map_source)

covering = create_covering(tile_labels, cfg.color_map)
covering.save(cfg.covering_target)

colorized_map = ImageChops.add(citymap, covering, 2.0)
colorized_map.save(cfg.colorized_map_target)