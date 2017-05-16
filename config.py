# -*- coding: utf-8 -*-

# Общие настройки

map_source = "./training/map.jpg"
tiles = "./training/tiles"

colorized_map_target = "./training/colorized_map.jpg"
covering_target = "./training/covering.png"

square_size = 150 		# Размер квадрата из файла, в пикселях
training_count = 150 	# Cколько примеров использовать в обучении


classes = { 
	'0': 'Жилой район', 
	'1': 'Индивидуальная застройка', 
	'2': 'Промышленный сектор', 
    '3': 'Водоохранная зона', 
    '4': 'Зеленая зона'
}

# Какими цветами раскрашивать каждый класс
color_map = { 
	0: 'red', 
	1: 'pink', 
	2: 'gray', 
	3: 'blue', 
	4: 'green'
}
