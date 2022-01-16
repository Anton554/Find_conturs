import numpy as np
import cv2 as cv
import os


def img_show(win_name, img):
    cv.imshow('window_name', img)
    cv.waitKey(0)


def conv_img(img):
    # Обрезка до середины
    # print(f'{img.shape=}')
    # x0 = int(img.shape[1] / 3)
    # y0 = int(img.shape[0] / 3)
    # img = img[y0:y0 + y0, x0:x0 + x0, :]
    # Превращаем в черно-белое изображение
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Умное распознование
    edges = cv.Canny(img_gray, 100, 200)
    # Находим нужный контур
    contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contour = max(contours, key=len)
    # Координаты квадрата цифры
    x, y, w, h = cv.boundingRect(contour)
    # Обрезаем изображение
    crop_img = img[y:y + h, x:x + w]
    # Инвертироваие цвета
    crop_img = 255 - crop_img
    crop_img = cv.resize(crop_img, (65, 65), interpolation=cv.INTER_AREA)
    # Добавление отступов
    crop_img = cv.copyMakeBorder(crop_img, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=[0, 0, 0, 0])
    crop_img = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)
    # Убираем серый цвет
    crop_img = cv.threshold(crop_img, 140, 255, cv.THRESH_BINARY)[1]
    # Утолщение
    kernel = np.ones((3, 3), np.uint8)
    crop_img = cv.dilate(crop_img, kernel, iterations=1)
    # Размытие по гаусу
    crop_img = cv.GaussianBlur(crop_img, (3, 3), cv.BORDER_DEFAULT)
    # Сжатие до 28, 28
    crop_img = cv.resize(crop_img, (28, 28))
    return img_gray, crop_img


def pars_img(class_num: str, img_name: str):
    np_arr = cv.imread(img_name)
    img_gray, img = conv_img(np_arr)
    save_file(class_num, img_gray, sub='raw')
    ph = save_file(class_num, img, sub='fin')
    return ph


def save_file(cl, img, sub='raw'):
    # def save_file():
    """ Сохраняет фото в каталоге

    :param img: np_arr
    :param :sub:  подкаталог
    :return:
    """
    # Ищем макисмальный суффикс
    ls_file = os.listdir(f'./img/{sub}')
    if len(ls_file) == 0:
        max_sufix = 0
    else:
        max_sufix = max([int(s.split('.')[0].split('_')[1]) for s in ls_file]) + 1
    cv.imwrite(f'./img/{sub}/{cl}-img_{max_sufix}.png', img)
    return f'C:/Projects/IT/Python/Find_conturs/img/{sub}/{cl}-img_{max_sufix}.png'


if __name__ == '__main__':
    # save_file()
    # np_arr = cv.imread('./2.jpg')
    # np_arr = cv.imread('./img/raw/6-2.jpg')
    # np_arr = cv.imread('./img/raw/6-img_154.png')
    np_arr = cv.imread('./imgRandomRotation330.png')
    conv_img(np_arr)

