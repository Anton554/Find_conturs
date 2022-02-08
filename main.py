import asyncio

import torch

import setup
import logging
from handlers import telebot
from aiogram import Bot, Dispatcher
from aiogram.contrib.fsm_storage.memory import MemoryStorage
import os

# Загрузка модели нейронной сети Yolov5_m
mode_yolo = torch.hub.load('ultralytics/yolov5', 'custom', './net/yolov5_m_300ep.pt')

bot = Bot(token=setup.token)
dp = Dispatcher(bot, storage=MemoryStorage())
dir_prog = os.path.dirname(os.path.abspath(__file__))


async def main():
    logging.basicConfig(level=logging.INFO)

    telebot.register_handlers(dp)
    telebot.register_handlers_final(dp)

    await dp.start_polling()


if __name__ == '__main__':
    asyncio.run(main())
