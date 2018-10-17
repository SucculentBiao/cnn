import numpy as np
import random

def parse(file_name):
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
