from multiprocessing import Manager
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
import copy
import json
import multiprocessing
import os
import pathlib
import random 

def read_jsonl(jsonl_file, limit=1e+8):
    output = []
    with open(jsonl_file,'r') as f:
        for idx, line in enumerate(f):
            output.append(json.loads(line.strip()))
            if idx == limit:
                break
    return output

def text_tokenizer(manager_dict, manager_list, batch, sep_left='[unused1]', sep_right='[unused2]', gold_world='wikipedia', max_length=300):
    nlp = English()
    tokenizer = nlp.Defaults.create_tokenizer(nlp)
    apply_tokenization = lambda x: [ doc.text for doc in tokenizer(x)]
    for item in batch:
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

def main():
    mentions_source_file = '/home/repositories/Zero-Shot-Entity-Linking/preprocessing/mentions.jsonl'
    mentions_target_folder = '/home/repositories/Zero-Shot-Entity-Linking/data/mentions_split_by_world/wikipedia'
    pathlib.Path(mentions_target_folder).mkdir(parents=True, exist_ok=True)

    mentions = read_jsonl(mentions_source_file, limit=1e+8)
    mentions = random.choices(mentions, k=1e+5)
    chunk_size = 1e+3
    mentions = [mentions[x:x+chunk_size] for x in range(0, len(mentions), chunk_size)]
    
    with Manager() as manager:
    
        manager_list = manager.list()
        arguments = [ (manager_list, batch) for batch in mentions ]    
    
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            pool.starmap(text_tokenizer, arguments)
    
        output_dict = {}
        for idx, item in enumerate(manager_list):
            output_dict[str(idx)] = copy.deepcopy(item)
        
        output_save_path = os.path.join(mentions_target_folder, 'mentions.json')
        with open(output_save_path,'w+') as f:
            json.dump(output_dict, f)

if __name__ == '__main__':
    main()