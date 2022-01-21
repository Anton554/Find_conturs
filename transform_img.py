"""
Создает множество изображений на основе картинок находящихся в папке с использованием трансформера
"""
import numpy as np
import cv2 as cv
import torch
from PIL import Image
from torchvision import transforms
from img_proces import save_file, conv_img


def my_fn(image, degr, path, num='no_first'):
    # Если изображение не первое
    if num == 'no_first':

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225]),
                                        transforms.RandomRotation((degr, degr), expand=True)])
        # Загружаем изображение
        img = Image.open(image)
        # Переводим изображение в numpy
        arr_np = np.array(img)
        # Сохраням временный файл
        cv.imwrite(f'./img/learn/img{num}{degr}.png', arr_np)
        # Умное обрезание изображения
        arr_np = conv_img2(cv.imread(f'./img/learn/img{num}{degr}.png'))
        # Переводим numpy в тензор
        tn = torch.from_numpy(arr_np)
        # Добавляем 2 канала и меняем местави слои
        tn = tn.expand(3, 65, 65).permute(1, 2, 0)
        # Переводим изображение в numpy
        arr_np = tn.numpy()
        # Переводим изображение в тензор и меняем слои
        demo_img = transform(arr_np).permute(1, 2, 0)
        # Переделываем тензор в numpy
        arr_np = demo_img.numpy()
        # Размытие по гаусу
        arr_np = cv.GaussianBlur(arr_np, (9, 9), cv.BORDER_DEFAULT)
        # Перевод в чероно-белый
        arr_np = cv.cvtColor(arr_np, cv.COLOR_BGR2GRAY)
        # Добавление рамки
        arr_np = cv.copyMakeBorder(arr_np, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=[0, 0, 0, 0])
        # Сжатие до 28, 28
        arr_np = cv.resize(arr_np, (28, 28))
        # Сохранение
        save_file(path, 255 * arr_np, f'learn/{path}')
    else:
        # Загружаем изображение
        img = Image.open(image)
        # Переделываем изображение в numpy
        arr_np = np.array(img)
        # Сохраням временный файл
        cv.imwrite(f'./img/learn/img{num}{degr}.png', arr_np)
        # Умное обрезание изображения
        _, arr_np = conv_img(cv.imread(f'./img/learn/img{num}{degr}.png'))
        # Сохранение
        save_file(path, arr_np, f'learn/{path}')


def conv_img2(img):
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
    crop_img = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)
    # Убираем серый цвет
    crop_img = cv.threshold(crop_img, 140, 255, cv.THRESH_BINARY)[1]
    # Утолщение
    kernel = np.ones((3, 3), np.uint8)
    crop_img = cv.dilate(crop_img, kernel, iterations=1)
    # Размытие по гаусу
    crop_img = cv.GaussianBlur(crop_img, (3, 3), cv.BORDER_DEFAULT)
    # Утолщение
    kernel = np.ones((3, 3), np.uint8)
    crop_img = cv.dilate(crop_img, kernel, iterations=1)
    return crop_img


def my_fn2(img, path):
    # Передаем картинку и путь
    my_fn(img, 0, path, num='First', )
    my_fn(img, 30, path)
    my_fn(img, -30, path)
    my_fn(img, 50, path)
    my_fn(img, -50, path)


for el in range(3, 500):
    try:
        my_fn2(f'./img/raw/7-img_{el}.png', '7')
    except:
        pass
