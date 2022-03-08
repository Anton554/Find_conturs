"""
Модуль для обработки изображения
"""

import numpy as np
import cv2 as cv
import os
# import main
# from main import dir_prog

dir_prog = os.path.dirname(os.path.abspath(__file__))


def img_show(img, win_name='window_name'):
    cv.imshow(win_name, img)
    cv.waitKey(0)


def conv_img(img):
    """ Преобразует серое изобр. в чёрно-белое 28 x 28

    :param img: np_arr
    :return: np_arr(серое), np_arr(чёрно-белое 28 x 28)
    """
    h1, w1 = 0, 0
    # Обрезка до середины
    # print(f'{img.shape=}')
    x0 = int(img.shape[1] / 3)
    y0 = int(img.shape[0] / 3)
    # print(x0, y0)
    img = img[y0:y0 + y0, x0:x0 + x0, :]
    img_show(img, 'win_1')
    # Превращаем в черно-белое изображение
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Умное распознование
    edges = cv.Canny(img_gray, 100, 200)
    # Находим нужный контур
    contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contour = max(contours, key=len)
    # Координаты квадрата цифры
    x, y, w, h = cv.boundingRect(contour)
    cv.rectangle(img, (x-8, y-8), (x + w+8, y + h+8), (0, 255, 0), 2)
    # cv.drawContours(img, contours, -1, (255, 0, 0), 1, cv.LINE_AA, hierarchy, 1)
    # img_show(img, 'win_2')
    # Поправочный коофицент для отцентровки изображения
    if h > w:
        h1 = int((h-w)/2)
    else:
        w1 = int((w-h)/2)
    # Исправление контуров изображения
    x1 = y-w1-10
    y1 = y + w1+10+h
    x2 = x-h1-10
    y2 = x + w + h1+10

    while x0 <= x1:
        x1 -= 1
    while y0 <= y1:
        y1 -= 1

    while x2 < 0:
        x2 += 1
    while y2 < 0:
        y2 += 1

    while x1 < 0:
        x1 += 1
    while y1 < 0:
        y1 += 1
    # print(x, y, w, h, h1)
    # Обрезаем изображение
    crop_img = img[x1:y1, x2:y2]
    # img_show(crop_img, 'win_3')
    # Инвертироваие цвета
    crop_img = 255 - crop_img
    crop_img = cv.resize(crop_img, (65, 65), interpolation=cv.INTER_AREA)
    # img_show(crop_img, 'Invert color')
    # Добавление отступов
    crop_img = cv.copyMakeBorder(crop_img, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=[0, 0, 0, 0])
    crop_img = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)
    # img_show(crop_img, 'copyMakeBorder')
    # Убираем серый цвет
    _, crop_img = cv.threshold(crop_img, 120, 255, cv.THRESH_BINARY)
    # img_show(crop_img, 'Delete Gray')
    # Удаление шума
    # !!! При обработки бланков необходимо   kernel = np.ones((3,3),np.uint8)
    kernel = np.ones((2,2), np.uint8)
    crop_img = cv.morphologyEx(crop_img, cv.MORPH_OPEN, kernel)
    # img_show(crop_img, 'morphologyEx')
    # Утолщение
    kernel = np.ones((2, 2), np.uint8)
    crop_img = cv.dilate(crop_img, kernel, iterations=1)
    # img_show(crop_img, 'dilate')
    # Размытие по гаусу
    # Убрано т.к. уменьшает процент предсказания
    # crop_img = cv.GaussianBlur(crop_img, (3, 3), cv.BORDER_DEFAULT)
    # img_show(crop_img, 'win_8')
    # Сжатие до 28, 28
    crop_img = cv.resize(crop_img, (28, 28))
    # img_show(crop_img, 'win_9')
    return img_gray, crop_img


def conv_img_pdf(img):
    """ Преобразует в чёрно-белое 28 x 28

    :param img: np_arr
    :return: np_arr(чёрно-белое 28 x 28)
    """
    h1, w1 = 0, 0
    # img_show(img, 'win_1')
    # Превращаем в черно-белое изображение
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Умное распознование
    edges = cv.Canny(img_gray, 100, 200)
    # Находим нужный контур
    contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contour = max(contours, key=len)
    # Координаты квадрата цифры
    x, y, w, h = cv.boundingRect(contour)
    # cv.rectangle(img, (x-8, y-8), (x + w+8, y + h+8), (0, 255, 0), 2)
    # cv.drawContours(img, contours, -1, (255, 0, 0), 1, cv.LINE_AA, hierarchy, 1)
    # img_show(img, 'win_2')
    # img_show(crop_img, 'win_3')
    # Инвертироваие цвета
    crop_img = 255 - img
    crop_img = cv.resize(crop_img, (65, 65), interpolation=cv.INTER_AREA)
    # img_show(crop_img, 'Invert color')
    # Добавление отступов
    # crop_img = cv.copyMakeBorder(crop_img, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=[0, 0, 0, 0])
    # crop_img = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)
    # img_show(crop_img, 'copyMakeBorder')
    # Убираем серый цвет
    _, crop_img = cv.threshold(crop_img, 130, 255, cv.THRESH_BINARY)
    # img_show(crop_img, 'Delete Gray')
    # Удаление шума
    # !!! При обработки бланков необходимо   kernel = np.ones((3,3),np.uint8)
    kernel = np.ones((1,1),np.uint8)
    crop_img = cv.morphologyEx(crop_img, cv.MORPH_OPEN, kernel)
    # img_show(crop_img, 'morphologyEx')
    # Утолщение
    kernel = np.ones((3, 3), np.uint8)
    crop_img = cv.dilate(crop_img, kernel, iterations=1)
    # img_show(crop_img, 'dilate')
    # Размытие по гаусу
    # Убрано т.к. уменьшает процент предсказания
    # crop_img = cv.GaussianBlur(crop_img, (3, 3), cv.BORDER_DEFAULT)
    # img_show(crop_img, 'win_8')
    # Сжатие до 28, 28
    crop_img = cv.resize(crop_img, (28, 28))
    # img_show(crop_img, 'win_9')
    return crop_img


def pars_img(class_num: str, img_name: str):
    """ Cохраняем фото в папке 'raw'(серое) и в папке 'fin' картинку 28 x 28 ч.б.

    Возвращаем полный путь к файлу в папке 'fin'
    """
    np_arr = cv.imread(img_name)
    img_gray, img = conv_img(np_arr)
    ph_raw = save_file(class_num, img_gray, sub='raw')
    ph_fin = save_file(class_num, img, sub='fin')
    return ph_raw, ph_fin


def save_file(cl, img, sub='raw'):
    """ Сохраняет фото в каталоге sub

    :param img: np_arr
    :param :sub:  подкаталог
    :return: полное имя сохранённого файла
    """
    # Ищем макисмальный суффикс
    ls_file = os.listdir(f'{dir_prog}/img/{sub}')
    if len(ls_file) == 0:
        max_sufix = 0
    else:
        max_sufix = max([int(s.split('.')[0].split('_')[1]) for s in ls_file]) + 1
    cv.imwrite(f'{dir_prog}/img/{sub}/{cl}-img_{max_sufix}.png', img)
    return f'{dir_prog}/img/{sub}/{cl}-img_{max_sufix}.png'


if __name__ == '__main__':
    # save_file()
    # np_arr = cv.imread('./test1.png')
    # np_arr = cv.imread('./img/raw/6-2.jpg')
    # np_arr = cv.imread('./img/raw/6-img_154.png')
    np_arr = cv.imread('./2qQCeWW2tHw.jpg')
    # np_arr = cv.imread('./wdH3zsiqUi8.jpg')
    conv_img(np_arr)

