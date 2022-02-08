"""
Модуль для распознавания PDF бланков ответов ЕГЭ
"""
import fitz
import os
import cv2 as cv
import tempfile
from main import mode_yolo


def png2jpg(filename):
    """ Конвертирует файл .png --> .jpg

    :param filename: Путь до файла .png
    :return: file.jpg
    """
    portion = os.path.splitext(filename)
    img = cv.imread(filename)
    img = cv.cvtColor(img, cv.COLOR_RGBA2BGRA)
    newname = portion[0] + ".jpg"
    cv.imwrite(newname, img)
    return newname


def exp_img(path_img):
    """ Извлекает картинки из PDF в ./img_from_pdf

    :param path_img: Путь до PDF
    :return: file.png
    """
    # ph = Path('./tmp')
    # ph.mkdir(parents=True, exist_ok=True)
    # Открываем pdf документ
    pdf_document = fitz.open(path_img)
    # список изображений, используемых на странице
    images = pdf_document.get_page_images(0)
    xref = images[0][0]
    pix = fitz.Pixmap(pdf_document, xref)
    # создаем временный файл
    path = tempfile.mktemp(suffix='.png', dir='tmp')
    pix.save(path)
    return path


def detect_blank(img_name):
    """ Распознавание изображения моделью Yolov5_m
    """
    results = mode_yolo(img_name, size=720)
    # Отрисовка контуров найденых цифр
    np_arr = cv.imread(img_name)
    cn_img = results.xyxy[0].size()[0]
    for n in range(cn_img):
        x1 = int(results.xyxy[0][n:n + 1, :][0][0].item())
        y1 = int(results.xyxy[0][n:n + 1, :][0][1].item())
        x2 = int(results.xyxy[0][n:n + 1, :][0][2].item())
        y2 = int(results.xyxy[0][n:n + 1, :][0][3].item())
        # pred = round(results.xyxy[0][n:n + 1, :][0][4].item() * 100, 2)
        # cls = int(results.xyxy[0][n:n + 1, :][0][5].item())
        # print(f"{x1=} {y1=} {x2=} {y2=} {pred=}% {cls=}")
        cv.rectangle(np_arr, (x1, y1), (x2, y2), (255, 255, 0), 1)
    result_png = tempfile.mktemp(suffix='.png', dir='tmp')
    cv.imwrite(result_png, np_arr)
    return result_png


def del_tmpfile(filename, typ):
    """ Удаляем временные файлы
    """
    portion = os.path.splitext(filename)
    os.remove(f'{portion[0]}.{typ}')



def start_det(pdf_name):
    img_name = exp_img(pdf_name)
    img_name = png2jpg(img_name)
    result_png = detect_blank(img_name)
    del_tmpfile(img_name, 'jpg')
    del_tmpfile(img_name, 'png')
    del_tmpfile(pdf_name, 'pdf')
    return result_png



if __name__ == '__main__':
    img_name = exp_img('./bilet_3.pdf')
    img_name = png2jpg(img_name)
    detect_blank(img_name)
    del_tmpfile(img_name)
