from PIL import Image
import config as cnf
import numpy as np

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def file_name(x,y):
	print(cnf.tiles+"/tile-%d-%d.png"%(x,y))
	return cnf.tiles+"/tile-%d-%d.png"%(x,y)

def load_tile(x,y):
	return load_image(file_name(x,y))