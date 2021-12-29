import numpy as np
import cv2 as cv
import os

def img_show(win_name, img):
    cv.imshow('window_name', img)
    cv.waitKey(0)


def conv_img(img_gray):
    img_gray = img_gray[:, :, :1]
    print(f'{img_gray.shape=}')
    # Открытие (Эрозия -> Расширение)
    kernel = np.ones((5, 5),np.uint8)
    logyEx = cv.morphologyEx(img_gray, cv.MORPH_OPEN, kernel)
    # Глобальная пороговая обработка
    ret, thresh = cv.threshold(logyEx, 150, 255, 0)
    # Нахождение контуров (для заливки)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Заливка контуров
    cv.drawContours(thresh, contours[1:], -1, (0,255,0), 3)
    # Размытие
    thresh = cv.GaussianBlur (thresh, (3,3), 0)

    # Нахождение контуров (для вырезания)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Нахождение максимального по площади контура
    ls_cont = []
    for cnt in contours[1:]:
        x, y, w, h = cv.boundingRect(cnt)
        # z = (x+y)-(w+h)
        z = (w-x)*(h-y)
        ls_cont.append([z, (x,y,w,h)])
    if len(ls_cont) == 0:
        print('Контур объекта не найден.')
        return None, None
    print(ls_cont)
    ls_cont = sorted(ls_cont)
    x, y, w, h  = ls_cont[0][1]
    x, y, w, h = x-5,y -5, w+10, h+10

    # Рисование контур
    cv.rectangle(thresh,(x,y),(x+w,y+h),(0,255,0),1)
    #
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv.dilate(thresh, kernel, iterations=1)
    #
    # # return thresh
    # Вырезание по контуру
    # thresh = thresh[y:y+h, x:x+w]
    #
    # # Инвертироваие цвета
    # crop_img = 255 - crop_img
    # return img_gray, crop_img
    img_show('win', thresh)


if __name__ == '__main__':
    img = cv.imread('./img/raw/2-img_0.png')
    conv_img(img)