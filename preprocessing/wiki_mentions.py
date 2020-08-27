import json
import os 
import copy 
import dataset
import sqlite3
from urllib.parse import unquote_plus
import multiprocessing
from multiprocessing import Manager
import sys

from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

import argparse
from html.parser import HTMLParser

class IdentifiableObject(object):
    @classmethod
    def get_class_name(cls):
        return cls.__name__

class Mention(IdentifiableObject):
    def __init__(self):
        self.entity = ''
        self.mention = ''

    def get_entity(self):
        return self.entity
    
    def get_mention(self):
        return self.mention
    
    def get_content(self):
        return self.mention
    
    def __repr__(self):
        return "Mention(entity: {0}, mention: {1})".format(self.entity, self.mention)

class Text(IdentifiableObject):
    def __init__(self, content):
        self.content = content
    
    def get_entity(self):
        return None
    
    def get_mention(self):
        return None
    
    def get_content(self):
        return self.content
    
    def __repr__(self):
        return "Text(content: {0})".format(self.content)

class WikipediaMentionsParser(HTMLParser):

    def __init__(self):
        super().__init__()
        self.data = []
        self.buffer = None

    def handle_starttag(self, tag, attrs):
        if tag and tag == 'a' and attrs and attrs[0][0] == 'href':
            self.buffer = Mention()
            self.buffer.entity = unquote_plus(attrs[0][1]).lower()

    def handle_endtag(self, tag):
        if self.buffer and self.buffer.get_class_name() == 'Mention':
            self.data.append(copy.deepcopy(self.buffer))
            self.buffer = None

    def handle_data(self, data):
        if self.buffer and self.buffer.get_class_name() == 'Mention':
            self.buffer.mention = data
        else:
            self.data.append(Text(data))



def jsonl_batch_reader(folder, batch_size=1e+5, num_batch=None):
    if not num_batch:
        num_batch = multiprocessing.cpu_count()
    files = [ os.path.join(folder,x) for x in os.listdir(folder) ]
    batch_group = []
    batch = []
    for filename in files:
        with open(filename,'r') as f:
            for line in f:
                batch.append(json.loads(line.strip()))
                if len(batch) == batch_size:
                    batch_group.append(copy.deepcopy(batch))
                    batch = []
                    if len(batch_group) == num_batch:
                        yield batch_group
                        batch_group = []

def convert_to_text(data, max_length=300, orientation='left'):
    content = " ".join([ x.get_content() for x in data])
    if orientation == 'left':
        return content[len(content)-max_length:len(content)]
    elif orientation == 'right':
        return content[0:max_length]

def get_mentions_and_context(data, max_length=300):
    for idx, item in enumerate(data):
        if item.get_class_name() == 'Mention':
            context_left = convert_to_text(data[0:idx], orientation='left', max_length=max_length)
            context_right = convert_to_text(data[idx+1:len(data)], orientation='right', max_length=max_length)
            row = {
                "context_left": context_left,
                "context_right": context_right,
                "mention": item.mention,
                "entity": item.entity
            }
            yield row
        else:
            continue

def text_parser(batch, max_length=300):
    for item in batch:
        parser = WikipediaMentionsParser()
        parser.feed(item['text'])
        for item in get_mentions_and_context(parser.data, max_length=max_length):
            yield item

def text_linker(manager_dict, batch, max_length=300):
    for item in text_parser(batch, max_length=max_length):
        try:
            gold_dui = manager_dict[item["entity"]]
            row = {
                "context_left": item["context_left"],
                "context_right": item["context_right"],
                "mention": item["mention"],
                "gold_dui": gold_dui
            }
            yield row
        except:
            continue

def text_tokenizer(manager_dict, manager_list, batch, sep_left='[unused1]', sep_right='[unused2]', gold_world='wikipedia', max_length=300):
    nlp = English()
    tokenizer = nlp.Defaults.create_tokenizer(nlp)
    apply_tokenization = lambda x: [ doc.text for doc in tokenizer(x)]
    for item in text_linker(manager_dict, batch, max_length=max_length):
        context_left = apply_tokenization(item['context_left'])
        context_right = apply_tokenization(item['context_right'])
        mention = apply_tokenization(item['mention'])
        anchored_context = context_left + [sep_right] + mention + [sep_left] + context_right
        anchored_context = [ x for x in anchored_context if x.strip() and x.strip('\n') ]
        row = {
            "raw_mention": item['mention'],
            "gold_dui": item['gold_dui'],
            "gold_world": gold_world,
            "anchored_context": anchored_context
        }
        manager_list.append(row)

def load_json(fname):
    with open(fname,'r') as f:
        return json.load(f)

def db_commit(manager_list, connection='sqlite:///mentions.db', table='mentions'):
    db = dataset.connect(connection)
    db.begin()
    for item in manager_list:
        try:
            db[table].insert(dict(
                    raw_mention=item['raw_mention'],
                    gold_dui=item['gold_dui'],
                    gold_world=item['gold_world'],
                    anchored_context=item['anchored_context']
                ))
            db.commit()
        except:
            db.rollback()

def main():

    wikipedia_mentions_folder = '/home/datasets/entity_linking/BLINK/data/wikipedia/wikimentions/AA'
    title2dui_path = '/home/repositories/Zero-Shot-Entity-Linking/data/worlds/wikipedia/title2dui.json'
    
    with Manager() as manager:
        manager_dict = manager.dict()
        manager_list = manager.list()

        with open(title2dui_path,'r') as f:
            title2dui_dict = json.load(f)

        manager_dict.update(title2dui_dict)

        for batch_group in jsonl_batch_reader(wikipedia_mentions_folder):
            arguments = [ (manager_dict, manager_list, batch) for batch in batch_group ]
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                pool.starmap(text_tokenizer, arguments)
            db_commit(manager_list)
            manager_list = manager.list()

if __name__ == '__main__':
    main()