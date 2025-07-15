import pandas as pd
import json 
import re

def row_to_sample(row: str): 
    prompt = f'Тема: автор рекламирует {row['Товар']}\n Локация и события: cтудия\n Триггеры товара для раскрытия: {row['Триггеры товара для раскрытия']}'
    response = f'Хук: {row['Хук']}\n Содержание: {row['Удержание до 15 сек.']}\n {row['Содержание']}\n Призыв к зрителю: {row['CTA']}'
    return {"prompt": prompt, "response": response}


def row_to_sample_case5(row: str): 
    prompt = f'Тема: автор рекламирует {row['Товар']}\n Локация и события: студия\n Триггеры товара для раскрытия: без триггеров'
    response = f'Хук: распаковка товара\n Содержание: {row['Содержание']}\n Призыв к зрителю: {row['CTA']}'
    return {"prompt": prompt, "response": response}


def row_to_sample_case6(row: str): 
    prompt = f'Тема: автор рекламирует {row['Товар']}\n Локация и события: студия\n Триггеры товара для раскрытия: {row['Триггеры товара для раскрытия']}'
    response = f'Хук: распаковка товара\n Содержание: {row['Содержание']}\n Призыв к зрителю: Артикул оставлен в комментариях'
    return {"prompt": prompt, "response": response}


def row_to_sample_caseYasno(row: str): 
    prompt = f"Тема: {row['Тема'].split('.')[1]}\n Локация и события: {row['Локации']}\n Триггеры товара для раскрытия: Промокод на скидку на 20% на первую сессию в 'Ясно'"
    response = f'Хук: {row['Хук']}\n Содержание: {row['Содержание']}\n Призыв к зрителю: {row['CTA']}'
    return {"prompt": prompt, "response": response}


def split_to_sample(text):
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
            data = [row_to_sample_case5(row) for _, row in df.iterrows()] 
            with open(f'resources/{file_name}.json', mode='w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return
        case 'bloggers6':
            df = df.drop(['№', 'Блогер', 'Формат','Надпись на обложке', 'CTA'], axis=1)
            df = df.dropna()
            data = [row_to_sample_case6(row) for _, row in df.iterrows()] 
            with open(f'resources/{file_name}.json', mode='w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return
        case 'bloggers7':
            df = df.drop(['№', 'Блогер', 'Формат','Надпись на обложке', 'CTA'], axis=1)
            df = df.dropna()
            data = [row_to_sample_case6(row) for _, row in df.iterrows()] 
            with open(f'resources/{file_name}.json', mode='w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return
        case 'bloggers8':
            df = df.drop(['№', 'Блогер', 'Формат','Надпись на обложке', 'CTA'], axis=1)
            df = df.dropna()
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


if __name__ == "__main__":
    # convert_csv_to_json('bloggers2')
    # convert_csv_to_json('bloggers3')
    # convert_csv_to_json('bloggers4')
    # convert_csv_to_json('bloggers5')
    # convert_csv_to_json('bloggers6')
    # convert_csv_to_json('bloggers7')
    # convert_csv_to_json('bloggers8')
    convert_csv_to_json('Yasno')