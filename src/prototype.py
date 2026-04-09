'''
Мы реализуем простую, но эффективную логику:
Признаки: Позиционное кодирование (каждая буква на своей позиции — отдельный признак).
Модель: Логистическая регрессия (быстрая, понятная, хороший бейзлайн).
Валидация: Разделим тренировочные данные, чтобы оценить качество до отправки ответа.
'''

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# --- КОНФИГУРАЦИЯ ---
DATA_RAW = os.path.join('data', 'raw')
DATA_PROCESSED = os.path.join('data', 'processed')
MODELS_DIR = 'models'
ANSWERS_DIR = os.path.join('data', 'answers')

# Создаем папки, если нет
os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ANSWERS_DIR, exist_ok=True)

def load_data():
    train = pd.read_csv(os.path.join(DATA_RAW, 'train_runes.csv'))
    test = pd.read_csv(os.path.join(DATA_RAW, 'test_runes.csv'))
    return train, test

def extract_features(df, is_train=True):
    """
    Превращаем строки 'rune' в числовые признаки.
    Используем One-Hot Encoding для каждой позиции.
    """
    # Создаем копию, чтобы не менять исходный DataFrame
    data = df.copy()
    
    # Для каждой из 5 позиций создаем признаки
    for i in range(5):
        # Извлекаем символ на позиции i
        col_name = f'char_{i}'
        data[col_name] = data['rune'].str[i]
    
    # Применяем One-Hot Encoding (get_dummies превращает буквы в 0/1)
    # Например, char_0_a, char_0_b, char_1_a...
    features = pd.get_dummies(data, columns=[f'char_{i}' for i in range(5)])
    
    # Удаляем исходную колонку rune и целевую (если есть)
    cols_to_drop = ['rune']
    if 'spell' in features.columns:
        cols_to_drop.append('spell')
        
    X = features.drop(columns=cols_to_drop, errors='ignore')
    return X

def main():
    print("1. Загрузка данных...")
    train, test = load_data()
    
    print("2. Извлечение признаков...")
    # X - признаки, y - целевая переменная
    X = extract_features(train)
    y = train['spell']
    
    # Для теста применяем ту же логику
    # Важно: get_dummies может создать разные колонки в train и test,
    # если в тесте встретились буквы, которых не было в тренировке.
    # В прототипе упростим: просто преобразуем тест.
    X_test_full = extract_features(test)
    
    # Приводим тест к тем же колонкам, что и трейн (заполняем отсутствующие нулями)
    X_test_full = X_test_full.reindex(columns=X.columns, fill_value=0)

    print("3. Разделение на Train/Valid (для проверки качества)...")
    # 80% на обучение, 20% на проверку. stratify сохраняет баланс классов
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("4. Обучение модели...")
    # class_weight='balanced' поможет, если классов 0 и 1 неравномерно
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)

    print("5. Валидация...")
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    print(f"   Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(f"   F1 Score: {f1_score(y_val, y_pred):.4f}")
    print(f"   ROC-AUC:  {roc_auc_score(y_val, y_pred_proba):.4f}")

    print("6. Финальное обучение на всех данных...")
    model.fit(X, y)

    print("7. Предсказание на тесте...")
    test_predictions = model.predict(X_test_full)
    
    # Формирование ответа
    answer_df = pd.DataFrame({
        'rune': test['rune'].values,
        'spell': test_predictions
    })
    
    # Сохранение
    output_path = os.path.join(ANSWERS_DIR, 'answers.csv')
    answer_df.to_csv(output_path, index=False)
    print(f"   Готово! Файл сохранен: {output_path}")

if __name__ == '__main__':
    main()