from PIL import Image
import numpy as np

def tranform(imageUrl):
    img= Image.open(imageUrl).convert('L')
    if img.size[0] != 28 or img.size[1] != 28:
        img = img.resize((28, 28))
    arr = []

    for i in range(28):
        for j in range(28):
            pixel = 1.0 - float(img.getpixel((j, i))) / 255.0
            arr.append(pixel)
    arr1 = np.array(arr).reshape((1, 784))
    return arr1.astype("float32")