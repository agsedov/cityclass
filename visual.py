import numpy as np
import keras
import config as cnf
from images import load_tile
import matplotlib.pyplot as plt
from map_utils import TileDataProvider
from model import init_model
import sys

model = init_model()
model.load_weights('./model_weights.h5')
tlp = TileDataProvider(cnf.map_source, cnf.square_size)
image_width, image_height = tlp.size()
i = 0
j = 0
max_i = int(np.floor(image_height/cnf.square_size))
max_j = int(np.floor(image_width/cnf.square_size))
model = init_model()
model.load_weights('./model_weights.h5')
tlp = TileDataProvider(cnf.map_source, cnf.square_size)

def press(event):
    global i,j
    print(i,j)

    if event.key == 'left' and j > 0:
        j = j - 1
    if event.key == 'right' and j < max_j:
        j = j + 1
    if event.key == 'up' and i > 0:
        i = i - 1
    if event.key == 'down' and i < max_i:
        i = i + 1
    wind(i,j)

def wind(i,j):
    global image, fig
    tile = tlp.load_tile_data(i,j)
    print(cnf.classes[str(model.predict_classes(np.array([tile]))[0])])
    image.set_data(tile.astype(np.uint8))
    fig.canvas.draw()

tile = tlp.load_tile_data(i,j)
print(cnf.classes[str(model.predict_classes(np.array([tile]))[0])])
fig, ax = plt.subplots(1, 1)
fig.canvas.mpl_connect('key_press_event', press)
image = plt.imshow(tile.astype(np.uint8))

wind(i,j)
plt.show()