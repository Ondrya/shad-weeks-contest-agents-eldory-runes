# Этот скрипт не участвует в обучении, он нужен только вам для анализа.

import pandas as pd
import os

# Пути к данным
DATA_RAW = os.path.join('data', 'raw')
TRAIN_PATH = os.path.join(DATA_RAW, 'train_runes.csv')
TEST_PATH = os.path.join(DATA_RAW, 'test_runes.csv')
EXAMPLE_PATH = os.path.join(DATA_RAW, 'example.csv')

def main():
    print("=== Загрузка данных ===")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    example = pd.read_csv(EXAMPLE_PATH)

    print("\n=== Тренировочные данные ===")
    print(f"Размер: {train.shape}")
    print(f"Колонки: {train.columns.tolist()}")
    print(train.head())
    
    # Важно: проверка баланса классов
    print("\n=== Распределение классов (spell) ===")
    print(train['spell'].value_counts())
    print(f"Доля единиц: {train['spell'].mean():.2%}")

    print("\n=== Тестовые данные ===")
    print(f"Размер: {test.shape}")
    print(test.head())

    print("\n=== Пример ответа ===")
    print(f"Колонки: {example.columns.tolist()}")
    print(example.head())

    # Проверка длины строк
    print("\n=== Проверка длины рун ===")
    train_len = train['rune'].str.len().unique()
    test_len = test['rune'].str.len().unique()
    print(f"Уникальные длины в train: {train_len}")
    print(f"Уникальные длины в test: {test_len}")

if __name__ == '__main__':
    main()