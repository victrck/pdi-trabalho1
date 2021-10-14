import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("img.jpg")


def view(img):
    plt.imshow(img)
    plt.show()
    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()


def log_transformation(img):
    num = float(input("Digite o valor a se calculado o log(>0):"))
    c = 255 / (np.log(num))
    log_img = np.array(c * (np.log(img + 1)), dtype=np.uint8)
    view(log_img)


def inv_transformation(img):
    inv_img = 255 - img
    view(inv_img)


def expo_transformation(img):
    expo = float(input("Digite o valor do expoente:"))
    expo_img = np.array(255 * (img / 255) ** expo, dtype="uint8")
    view(expo_img)


# view(img)
log_transformation(img)
expo_transformation(img)
# inv_transformation(img)
