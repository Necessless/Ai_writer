import pandas as pd
import json 
import re


def row_to_sample(row: str): 
    """Метод для 1-4 датасетов, форматирует каждый пример под sample для json"""
    return {
        "prompt": {
            "task": "Создай небольшой сценарий для рекламного вертикального видео (TikTok/Instagram Reels)",
            "theme": 'Реклама товара',
            "product":  row['Товар'],
            "location": 'съёмочная студия',
            "triggers": row['Триггеры товара для раскрытия']
        },  
        "response": {
            "hook": row['Хук'],
            "story": row['Содержание'],
            "cta": row['CTA']
        }      
    }


def row_to_sample_case5(row: str): 
    """Тот же самый метод, что и выше, но для 5-го датасета"""
    return {
        "prompt": {
            "task": "Создай небольшой сценарий для рекламного вертикального видео (TikTok/Instagram Reels)",
            "theme": 'Реклама товара',
            "product":  row['Товар'],
            "location": 'съёмочная студия',
            "triggers": 'хороший товар'
        },  
        "response": {
            "hook": row['Хук'],
            "story": row['Содержание'],
            "cta": row['CTA']
        }      
    }


def row_to_sample_case6(row: str):
    """Метод, форматирующий 6-ой датасет"""
    return {
        "prompt": {
            "task": "Создай небольшой сценарий для рекламного вертикального видео (TikTok/Instagram Reels)",
            "theme": 'Реклама товара',
            "product":  row['Товар'],
            "location": 'съёмочная студия',
            "triggers": row['Триггеры товара для раскрытия']
        },  
        "response": {
            "hook":  row['Хук'],
            "story": row['Содержание'],
            "cta": row['CTA']
        }      
    }


def row_to_sample_caseYasno(row: str): 
    """Метод, форматирующий датасет с примерами рекламы 'ЯСНО'"""
    return {
        "prompt": {
            "task": "Создай небольшой сценарий для рекламного вертикального видео (TikTok/Instagram Reels)",
            "theme": row['Тема'].split('.')[1],
            "product":  'психологический сервис ЯСНО',
            "location": row['Локации'],
            "triggers": 'промокод на 20% скидку в ЯСНО'
        },  
        "response": {
            "hook": row['Хук'],
            "story": row['Содержание'],
            "cta": row['CTA']
        }      
    }


def split_to_sample(text):
    """Метод, для разбиения одного столбца 'Содержимое' на 'CTA', 'Хук'"""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) >= 2:
        cta = " ".join(sentences[-1:]) 
        body = " ".join(sentences[1:-1])
        hook = sentences[0]
    else:
        cta = text
        body = ""
        hook = ""
    return pd.Series({"body": body, "CTA": cta, "Хук": hook})


def convert_csv_to_json(file_name):
    """Метод для конвертации всех датасетов из csv в json"""
    df = pd.read_csv(f'resources/{file_name}.csv')
    match file_name:
        case 'bloggers2':
            df = df.drop(['№', 'Блогер', 'Артикул', 'Ссылка на типовой сценарий'], axis=1)
        case 'bloggers3':
            df = df.drop(['Unnamed: 0', 'Блогер', 'Артикул', 'Ссылка на типовой сценарий'], axis=1)
        case 'bloggers4':
            df = df.drop(['№', 'Блогер', 'Артикул', 'Ссылка на типовой сценарий'], axis=1)
        case 'bloggers5':
            df = df.drop(['№', 'Блогер', 'Артикул', 'Формат','Надпись на обложке', 'Триггеры товара для раскрытия'], axis=1)
            df = df.dropna()
            df[["Содержание", "CTA", 'Хук']] = df["Содержание"].apply(split_to_sample)
            data = [row_to_sample_case5(row) for _, row in df.iterrows()] 
            with open(f'resources/{file_name}.json', mode='w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return
        case 'bloggers6':
            df = df.drop(['№', 'Блогер', 'Формат','Надпись на обложке', 'CTA'], axis=1)
            df = df.dropna()
            df[["Содержание", "CTA", 'Хук']] = df["Содержание"].apply(split_to_sample)
            data = [row_to_sample_case6(row) for _, row in df.iterrows()] 
            with open(f'resources/{file_name}.json', mode='w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return
        case 'bloggers7':
            df = df.drop(['№', 'Блогер', 'Формат','Надпись на обложке', 'CTA'], axis=1)
            df = df.dropna()
            df[["Содержание", "CTA", 'Хук']] = df["Содержание"].apply(split_to_sample)
            data = [row_to_sample_case6(row) for _, row in df.iterrows()] 
            with open(f'resources/{file_name}.json', mode='w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return
        case 'bloggers8':
            df = df.drop(['№', 'Блогер', 'Формат','Надпись на обложке', 'CTA'], axis=1)
            df = df.dropna()
            df[["Содержание", "CTA", 'Хук']] = df["Содержание"].apply(split_to_sample)
            data = [row_to_sample_case6(row) for _, row in df.iterrows()] 
            with open(f'resources/{file_name}.json', mode='w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return
        case 'Yasno':
            df = df.drop(['Ясно ', 'Unnamed: 3'], axis=1)
            df = df.drop(0, axis=0)
            df = df.rename(columns={"Unnamed: 1": "Тема", "Unnamed: 2": "Содержание"})
            df = df.dropna()
            df[["Содержание", "CTA", 'Хук']] = df["Содержание"].apply(split_to_sample)
            data = [row_to_sample_caseYasno(row) for _, row in df.iterrows()] 
            with open(f'resources/{file_name}.json', mode='w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return

    df = df.dropna()
    data = [row_to_sample(row) for _, row in df.iterrows()] 
    with open(f'resources/{file_name}.json', mode='w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def summarize_jsons():
    """Конкатенация всех json датасетов в один"""
    summarize_list = []
    for i in range(2,9):
        if i == 5: continue  # по итогам метрик, решил убрать 5-ый датасет за ненадобностью
        with open(f'resources/bloggers{i}.json', mode='r', encoding='utf-8') as f:
            samples = json.load(f)
            summarize_list.extend(samples)
    with open(f'resources/Yasno.json', mode='r', encoding='utf-8') as f:
        samples = json.load(f)
        summarize_list.extend(samples)
    with open('dataset/data.jsonl', mode='w', encoding='utf-8') as f:
        json.dump(summarize_list, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    convert_csv_to_json('bloggers2')
    convert_csv_to_json('bloggers3')
    convert_csv_to_json('bloggers4')
    convert_csv_to_json('bloggers5')
    convert_csv_to_json('bloggers6')
    convert_csv_to_json('bloggers7')
    convert_csv_to_json('bloggers8')
    convert_csv_to_json('Yasno')
    summarize_jsons()