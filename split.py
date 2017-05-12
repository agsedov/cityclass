import config as cnf
from PIL import Image

# Скриптик для того чтобы разбить картинку на квадратики

def split_to_tiles(save_path, input, size):
    im = Image.open(input)
    print("!!!1");
    imgwidth, imgheight = im.size

    for i in range(0,imgheight,size):
        for j in range(0,imgwidth,size):
            box = (j, i, j+size, i+size)
            a = im.crop(box)
            
            tile_address = save_path+"/tile-%d-%d.png"%(i/size,j/size)
            a.save(tile_address)
            

split_to_tiles(cnf.tiles, cnf.map_source, cnf.square_size)