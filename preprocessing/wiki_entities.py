wikipedia_source = '/home/datasets/entity_linking/BLINK/data/wikipedia/wiki/AA'
wikipedia_target = '/home/repositories/Zero-Shot-Entity-Linking/data/documents/wikipedia.json'
wikipedia_world_folder = '/home/repositories/Zero-Shot-Entity-Linking/data/worlds/wikipedia'

import os
import copy
import json
import pathlib

class Reader(object):

    def __init__(self, folder):
        self.folder = os.listdir(folder)

    def line_generator(self):
        for item in self.folder:
            with open(os.path.join(self.folder, item),'r') as f:
                for line in f:
                    yield json.loads(line.strip())
    
class WikipediaDumpReader(Reader):

    def __init__(self, folder):
        super().__init__(folder)
    
    def line_parser(self, trim_characters=300):
        for line in self.line_generator():
            row = copy.deepcopy(line)
            row['text'] = row['text'][0:trim_characters]
            yield row

def jsonl_append(row, fname):
    with open(fname,'a+') as f:
        print(json.dumps(row), file=f)

def write_wikipedia_document(wiki_source, wiki_target):
    open(wiki_target,'w+').close()
    wdr = WikipediaDumpReader(wiki_source)
    for line in wdr.line_parser():
        row = {
            "title": line["title"],
            "text": line["text"],
            "document_id": line["id"]
        }
        jsonl_append(row, wiki_target)

def write_wikipedia_world(wiki_target, wiki_world_folder):
    pathlib.Path(wiki_world_folder).mkdir(parents=True, exist_ok=True)

    dui2desc_path = os.path.join(wiki_world_folder, 'dui2desc.json')
    dui2idx_path = os.path.join(wiki_world_folder, 'dui2idx.json')
    dui2title_path = os.path.join(wiki_world_folder, 'dui2title.json')
    idx2dui_path = os.path.join(wiki_world_folder, 'idx2dui.json')

    title2dui_path = os.path.join(wiki_world_folder, 'title2dui.json')

    dui2desc_dct = {}
    dui2idx_dct = {}
    dui2title_dct = {}
    idx2dui_dct = {}

    title2dui_dct = {}

    with open(wiki_target,'r') as f:
        for idx, line in enumerate(f):
            row = json.loads(line.strip()) # title, text, document_id

            #dui2desc
            dui2desc_dct[row['document_id']] = row['text']

            #dui2idx
            dui2idx_dct[row['document_id']] = idx
            
            #dui2title
            dui2title_dct[row['document_id']] = row['title']

            #idx2dui
            idx2dui_dct[idx] = row['document_id']

            #title2dui
            title2dui_dct[row['title']] = row['document_id']
    
    save_json(dui2desc_dct, dui2desc_path)
    save_json(dui2idx_dct, dui2idx_path)
    save_json(dui2title_dct, dui2title_path)
    save_json(idx2dui_dct, idx2dui_path)

    save_json(title2dui_dct, title2dui_path)

def save_json(dct, path):
    with open(path,'w+') as f:
        json.dump(dct, f)

def main():

    pathlib.Path(os.path.split(wikipedia_target)[0]).mkdir(parents=True, exist_ok=True)
    pathlib.Path(wikipedia_world_folder).mkdir(parents=True, exist_ok=True)
    
    write_wikipedia_document(wikipedia_source, wikipedia_target)
    write_wikipedia_world(wikipedia_target, wikipedia_world_folder)

if __name__ == '__main__':
    main()