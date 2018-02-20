import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as pyplot

def array_to_image(arr, image_filename, channels = 1, size = 28, normalized = True):
    # For mnist, channels = 1, size = 28
    # For cifar-10, channels = 3, size = 32

    if normalized == True: n = 255.0
    else: n = 1.0
    rows = arr.shape[0]
    arr = arr.reshape(rows, channels, size, size)

    for index in range(rows):
        a = arr[index]
        if channel == 3:
            r = Image.fromarray(a[0] * n).convert('L')
            g = Image.fromarray(a[1] * n).convert('L')
            b = Image.fromarray(a[2] * n).convert('L')
        elif channel == 1:
            r = Image.fromarray(a[0] * n).convert('L') 
            g = r
            b = r

        image = Image.merge("RGB", (r, g, b))
        # show the image
        pyplot.imshow(image)
        pyplot.show()
        image.save(image_filename + "_" + str(index) + ".png", 'png')
