import numpy as np
import cv2 as cv
import os


def img_show(win_name, img):
    cv.imshow('window_name', img)
    cv.waitKey(0)


def conv_img(img):
    """

    :param img: np_arr
    :return:
    """
    # Изменение размер
    print(f'{img.shape=}')
    x0 = int(img.shape[1] / 3)
    y0 = int(img.shape[0] / 3)
    img = img[y0:y0 + y0, x0:x0 + x0, :]
    resized = cv.resize(img, (224, 224), interpolation=cv.INTER_AREA)
    # В серый
    img_gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
    # Открытие (Эрозия -> Расширение)
    kernel = np.ones((5, 5), np.uint8)
    logyEx = cv.morphologyEx(img_gray, cv.MORPH_OPEN, kernel)
    # Глобальная пороговая обработка
    ret, thresh = cv.threshold(logyEx, 150, 255, 0)
    # Нахождение контуров (для заливки)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Заливка контуров
    cv.drawContours(thresh, contours[1:], -1, (0, 255, 0), 3)
    # Размытие Гауса
    thresh = cv.GaussianBlur(thresh, (3, 3), 0)
    # Нахождение контуров (для вырезания)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Нахождение максимального по площади контура
    ls_cont = []
    for cnt in contours[1:]:
        x, y, w, h = cv.boundingRect(cnt)
        z = (x + y) - (w + h)
        ls_cont.append([z, (x, y, w, h)])
    if len(ls_cont) == 0:
        print('Контур объекта не найден.')
        return None, None
    print(ls_cont)
    ls_cont = sorted(ls_cont)
    x, y, w, h = ls_cont[0][1]
    # x, y, w, h = x, y - 5, w + 10, h + 10
    # Рисование контур
    # cv.rectangle(thresh,(x,y),(x+w,y+h),(0,255,0),1)
    # Утоншение
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv.dilate(thresh, kernel, iterations=1)
    # Вырезание по контуру
    crop_img = thresh[y:y + h, x:x + w]
    # Инвертироваие цвета
    crop_img = 255 - crop_img
    crop_img = cv.resize(crop_img, (65, 65), interpolation=cv.INTER_AREA)
    crop_img = cv.copyMakeBorder(crop_img, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=[0, 0, 0, 0])
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
    np_arr = cv.imread('./img/raw/2-img_1.png')
    img_gray, img = conv_img(np_arr)
    img_show('im2', img)
