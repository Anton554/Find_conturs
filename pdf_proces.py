"""
Модуль для распознавания PDF бланков ответов ЕГЭ
"""
import fitz
import os
import cv2 as cv
import tempfile

dir_prog = os.path.dirname(os.path.abspath(__file__))

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
    path = tempfile.mktemp(suffix='.png', dir=dir_prog + os.sep + 'tmp')
    pix.save(path)
    return path


def pdf2jpg(pdf_name):
    img_png = exp_img(pdf_name)
    img_jpg = png2jpg(img_png)
    os.remove(img_png)
    return img_jpg


if __name__ == '__main__':
    img_png = exp_img('./bilet_3.pdf')
    img_name = png2jpg(img_png)
    # detect_blank(img_name)

