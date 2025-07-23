import json
import os
import re
import rich

from rich.progress import Progress

def get_default_rich_progress():

    return Progress(rich.progress.TextColumn("[progress.description]{task.description}"),
                    rich.progress.BarColumn(),
                    rich.progress.TextColumn('({task.completed}/{task.total})'),
                    rich.progress.TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    rich.progress.TimeRemainingColumn(),
                    rich.progress.TimeElapsedColumn())

# https://rich.readthedocs.io/en/stable/logging.html
def get_default_rich_logger(name):
    import logging
    from rich.logging import RichHandler

    simple_format = '%(name)s - %(message)s'

    logging.basicConfig(
        level="NOTSET",
        format=simple_format,
        # datefmt="[%X]",
        handlers=[RichHandler()]
        # handlers=[RichHandler(rich_tracebacks=True)]
    )
    # log.error("[bold red blink]Server is shutting down![/]", extra={"markup": True})
    # log.error("123 will not be highlighted", extra={"highlighter": None})

    return logging.getLogger(name)

def save_to_json(data, file_path):
    if (os.path.dirname(file_path) != '') and (not os.path.exists(os.path.dirname(file_path))):
        os.makedirs(os.path.dirname(file_path))

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def read_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_to_flatten_json(data, file_path):
    if (os.path.dirname(file_path) != '') and (not os.path.exists(os.path.dirname(file_path))):
        os.makedirs(os.path.dirname(file_path))

    with open(file_path, 'w', encoding='utf-8') as f:
        for d in data:
            json_str = json.dumps(d, ensure_ascii=False)
            f.write(json_str + '\n')

def read_from_flatten_json(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def contains_only_punctuation(text):
    punctuation = r'[\u3000-\u303f\uff00-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65\u2018-\u2019\u201c-\u201d\.\。\,\，\;\；\:\：\!\！\?\？$\（$\）$$\〔$$\〕\{\【\}\】\-\——\…\'\"\‘\’\“\”\《\》\「\」\『\』\【\】\（\）]'
    return bool(re.fullmatch(f'^{punctuation}+$', text))

def document_to_flatten_json(documents):
    # from langchain_core.documents import Document
    data = []

    for doc in documents:
        info = {}
        
        info['context'] = doc.page_content.replace('\n', '')

        if 'summary' in doc.metadata:
            info['title'] = doc.metadata['summary']

        if 'chunk_id' in doc.metadata:
            info['id'] = doc.metadata['chunk_id']

        if 'content_type' in doc.metadata:
            info['type'] = doc.metadata['content_type']

        data.append(info)

    return data


def transcode_text(file_path, output_path, source='gbk', target='utf-8'):
    with open(file_path, 'rb') as f:
        content = f.read()
    content = content.decode(source).encode(target)
    
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    with open(output_path, 'wb') as f:
        f.write(content)



if __name__ == '__main__':
    test = '你好，世界！'
    test = ';；'
    test = '。，sdafdsaf'
    print(contains_only_punctuation(test))