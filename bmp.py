import numpy as np
from PIL import Image

def parse(file_name):
    img = Image.open(file_name)
    pixels = list(img.getdata())
    for i in range(len(pixels)):
        pixels[i] /= 255
    img.close()
    return pixels

def old_parse(file_name):
    img = open(file_name, 'rb')
    img.read(62)
    int_pixels = []
    pixels = []
    for i in range(98):
        int_pixels.append(int.from_bytes(img.read(1), byteorder='little'))
    img.close()

    for i in range(len(int_pixels)):
        s = bin(int_pixels[i])
        s = s[2:]
        for j in range(8 - len(s)):
            s = '0' + s
        for j in range(len(s)):
            pixels.append(float(s[j]))
    return pixels