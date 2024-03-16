# README in Russian:


## MNIST Нейронный  классификатор

Этот проект представляет собой простой нейронный классификатор, обученный на наборе данных MNIST для распознавания рукописных цифр.

## Структура проекта

- `main.py`: Основной файл, содержащий код для обучения моделей и их оценки.
- `models.py`: Файл, в котором определены архитектуры нейронных сетей (SimpleNN и DropoutBatchNormNN).
- `utils.py`: Файл с вспомогательными функциями, такими как функция для обучения с графиком и функция для оценки моделей.
- `README.md`: Файл, который вы сейчас читаете, содержащий описание проекта и инструкции по запуску.

## Зависимости

Проект используйет следующие библиотеки для работы:

- torch
- torchvision
- numpy
- matplotlib
- scikit-learn(sklearn)

## Запуск проекта

При запуске `main.py` будут обучаться две модели: SimpleNN и DropoutBatchNormNN, их лоссы будут отображены на графиках. После обучения моделей будут выведены их метрики (accuracy, precision, recall и f1score) на датасэте MNIST.

Для изменения количества эпох зайдите в файл utils.py и найдите там комментарий, где показывается что надо изменить.


============================

# README in English

## MNIST Neural Classifier

This project is a simple neural classifier trained on the MNIST dataset to recognize handwritten digits.

## Project Structure

- `main.py`: Main file containing the code to train the models and evaluate them.
- `models.py`: The file where the neural network architectures (SimpleNN and DropoutBatchNormNNN) are defined.
- `utils.py`: A file with auxiliary functions such as a function for training with a graph and a function for evaluating models.
- `README.md`: The file you are currently reading, containing a project description and startup instructions.

## Dependencies

The project uses the following libraries to run:

- torch
- torchvision
- numpy
- matplotlib
- scikit-learn(sklearn)

## Start the project

When `main.py` is started, two models will be trained: SimpleNN and DropoutBatchNormNNN, their lots will be displayed on the graphs. After training the models, their metrics (accuracy, precision, recall and f1score) will be displayed on the MNIST dataset.

To change the number of epochs, go to the utils.py file and find a comment there that shows what needs to be changed.
