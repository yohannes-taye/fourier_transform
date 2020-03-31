import numpy as np
from math import floor, ceil
from PIL import Image
from numpy import asarray
import math
import matplotlib.pyplot as plt

'''
A function to convert from an ndarray and save 
it to the specified location
'''
def save_image(img: np.ndarray, address):
    image = Image.fromarray(img)

    return image.save(address)


def get_window(full_image, location, window_size):
    """
    1. check if it goes out of bound 
    2. if not generate a blank window of requested size 
    3. iteratively map pixels in full image to window 
    """
    window = []
    window_r = []
    window_g = []
    window_b = []

    start_x = location[0] - 1
    start_y = location[1] - 1
    img_size_x, img_size_y, _ = full_image.shape
    for x in range(3):
        for y in range(3):
            try:
                if start_x + x < img_size_x and start_y + y < img_size_y:
                    r, g, b = full_image[start_x + x][start_y + y]
                    window_r.append(r)
                    window_g.append(g)
                    window_b.append(b)
            except IndexError:
                print("___________________")
                print("Index out of bound")
                print("___________________")
                print("location: " + str(location))
                print("image size: " + str(full_image.shape))

            # window.append(full_image[start_x + x][start_y + y])
            # print("Appending " + str(full_image[start_x + x][start_y + y]))
    return [window_r, window_g, window_b]


def iexp(n):
    return complex(math.cos(n), math.sin(n))


def get_fourier(location, image):
    row, col, _ = image.shape
    fourier_value = 0
    for x in range(row):
        for y in range(col):
            fourier_value += image[x][y][0] * iexp(-2 * math.pi * (
                (location[0] * x) / row + (location[1] * y) / col))
    return fourier_value
 
def apply_fourier(img):
    row, col, _ = image.shape
    new_image = np.zeros([row, col], dtype=np.uint8)
    plt.subplot(231),plt.imshow(img),plt.title('picture')
    img = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
    print(img)
    plt.subplot(232),plt.imshow(img,'gray'),plt.title('original')
    print("row: {}, col: {}".format(row, col))
    print("sample {}".format(image[0][0][1]))
    for x in range(row):
        for y in range(col):
            temp = get_fourier([x, y], image)
            new_image[x][y] = temp
            print("fourier_value at[{}, {}]: {}".format(x, y, temp))
    plt.subplot(233),plt.imshow(np.abs(new_image),'gray'),plt.title('fft2')
    # save_image(new_image, "./img/mar.jpg")
    log_fft2 = np.log(1 + np.abs(new_image))
    plt.subplot(235),plt.imshow(log_fft2,'gray'),plt.title('log_fft2')

    # log_shift2center = np.log(1 + np.abs(shift2center))
    # plt.subplot(236),plt.imshow(log_shift2center,'gray'),plt.title('log_shift2center')


    # print("fourier_value at[{}, {}]: {}".format(x, y, abs(fourier_value)))
    plt.show()

def remove_noise_median(image):
    row, col, _ = image.shape
    new_image = np.zeros([row, col, 3], dtype=np.uint8)
    for x in range(row):
        for y in range(col):
            window_r, window_g, window_b = get_window(image, [x, y], 3)
            median_rgb = [np.mean(window_r), np.mean(
                window_g), np.mean(window_b)]
            new_image[x][y] = median_rgb

            # print("Window recieved " + str(window))

    save_image(new_image, "./img/yesssss.jpg")


if __name__ == "__main__":
    image = plt.imread('./img/img2.jpg')
    x, y, _ = image.shape
    print(image[0][0])
    data = asarray(image)
    apply_fourier(image)
