import numpy as np
from os import system
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("img.jpg")
img2 = cv2.imread("img2.jpg")


def show_hist(img):
    h = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.figure()
    plt.title("Histograma")
    plt.xlabel("Intensidade")
    plt.ylabel("Número de Pixels")
    plt.plot(h)
    plt.xlim([0, 256])
    plt.show()


def transformation_views(img):
    plt.imshow(img)
    plt.show()
    show_hist(img)


def hist_colors(img):
    chanels = cv2.split(img)
    c = ("b", "g", "r")
    plt.figure()
    plt.title("Histograma Colorido")
    plt.xlabel("Intensidade")
    plt.ylabel("Número de Pixels")
    for (canal, cor) in zip(chanels, c):
        hist = cv2.calcHist([canal], [0], None, [256], [0, 256])
        plt.plot(hist, color=cor)
        plt.xlim([0, 256])
    plt.show()


def hist_equalizer(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_eq = cv2.equalizeHist(img)
    show_hist(h_eq)
    show_hist(img)


def modelsandfilters_view(new_img, img):
    try:
        (h, s, v) = cv2.split(new_img)
        zeros = np.zeros(new_img.shape[:2], dtype="uint8")
        cv2.imshow("Vermelho", cv2.merge([zeros, zeros, h]))
        cv2.imshow("Verde", cv2.merge([zeros, s, zeros]))
        cv2.imshow("Azul", cv2.merge([v, zeros, zeros]))
        cv2.imshow("Transformada", new_img)
        cv2.imshow("Original", img)
        cv2.waitKey(0)
    except:
        cv2.imshow("Transformada", new_img)
        cv2.imshow("Original", img)
        cv2.waitKey(0)


def log_transformation(img):
    num = float(input("Digite o valor a se calculado o log(>0):"))
    c = 255 / (np.log(num))
    log_img = np.array(c * (np.log(img + 1)), dtype=np.uint8)
    transformation_views(log_img)


def inv_transformation(img):
    inv_img = 255 - img
    transformation_views(inv_img)


def expo_transformation(img):
    expo = float(input("Digite o valor do expoente:"))
    expo_img = np.array(255 * (img / 255) ** expo, dtype="uint8")
    transformation_views(expo_img)


def rgb_to_hsv(img):
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    modelsandfilters_view(new_img, img)


def hsv_to_rgb(img):
    new_img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    modelsandfilters_view(new_img, img)


def rgb_to_gray(img):
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    modelsandfilters_view(new_img, img)


def median_filter(img):
    new_img = cv2.medianBlur(img, 5)
    modelsandfilters_view(new_img, img)


def avg_filter(img):
    new_img = cv2.blur(img, (5, 5))
    modelsandfilters_view(new_img, img)


def avg_pond_filter(img):
    new_img = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    modelsandfilters_view(new_img, img)


def main_menu():
    op = input(
        "Escolha uma opcao: \n \
                1 - Aplicar Transformações de Intensidade\n \
                2 - Opções relacionadas a Histograma\n \
                3 - Modelos de cor e Filtros Espaciais\n \
                Opção:"
    )
    return op


def menu_filtersandmodels(img):
    op = input(
        "Escolha uma opcao: \n \
                1 - Transformar para RGB para HSV\n \
                2 - Transformar para HSV para RGB\n \
                3 - Transformar para preto e branco\n \
                4 - filtro da mediana\n \
                5 - filtro da media\n \
                6 - filtro da media ponderada\n \
                Opção:"
    )
    if op == "1":
        rgb_to_hsv(img)
    elif op == "2":
        hsv_to_rgb(img)
    elif op == "3":
        rgb_to_gray(img)
    elif op == "4":
        median_filter(img)
    elif op == "5":
        avg_filter(img)
    elif op == "6":
        avg_pond_filter(img)


def transfor_menu(img):
    op = input(
        "Escolha uma opcao: \n \
                1 - Aplicar Transformações Logaritma\n \
                2 - Aplicar Transformações Exponencial\n \
                3 - Aplicar Transformações Inversa\n \
                Opção:"
    )
    if op == "1":
        log_transformation(img)
    elif op == "2":
        expo_transformation(img)
    elif op == "3":
        inv_transformation(img)


def hist_menu(img, img2):
    op = input(
        "Escolha uma opcao: \n \
                1 - Exibir\n \
                2 - Equalizar\n \
                3 - Especificar\n \
                4 - Comparar\n \
                Opção:"
    )
    if op == "1":
        transformation_views(img)
        hist_colors(img2)
    elif op == "2":
        hist_equalizer(img)
        hist_equalizer(img2)
    elif op == "3":
        transfor_menu(img)
    elif op == "4":
        transfor_menu(img)
        transformation_views(img)
        hist_equalizer(img)


op = main_menu()
system("clear")
if op == "1":
    transfor_menu(img)
elif op == "2":
    hist_menu(img, img2)
elif op == "3":
    menu_filtersandmodels(img2)
