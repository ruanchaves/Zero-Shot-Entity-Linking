import requests, json
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
nlp = English()
tokenizer = Tokenizer(nlp.vocab)
import os 


benchmarks_folder = '/home/datasets/entity_linking/BLINK/data/BLINK_benchmark'
logfile = './log.jsonl'

def dataset_reader(fname):
    with open(fname,'r') as f:
        for line in f:
            json_line = json.loads(line.strip())
            row = {
                'entry_id': json_line['Wikipedia_ID'],
                'context_left': json_line['context_left'],
                'context_right': json_line['context_right'],
                'mention': json_line['mention']
            }
            yield row

class MentionList(object):

    def __init__(self):
        self.nlp = English()
        self.tokenizer = self.nlp.Defaults.create_tokenizer(nlp)
        self.sep_left = "[unused1]"
        self.sep_right = "[unused2]"
        self.gold_world = "wikipedia"
        self.counter = 0
        self.request = {}

    def _tokenize(self, text):
        return [doc.text for doc in self.tokenizer(text)]
    
    def append_from_fields(self, context_left, mention, context_right, entry_id):

        cl = self._tokenize(context_left)
        cr = self._tokenize(context_right)
        m = self._tokenize(mention)
        anchored_context = cl + [self.sep_left] + m + [self.sep_right] + cr

        self.request[str(self.counter)] = {
            "raw_mention": mention,
            "gold_dui": entry_id,
            "gold_world": self.gold_world,
            "anchored_context": anchored_context, 
        }

        self.counter += 1

    def __repr__(self):
        return json.dumps(self.request, indent=4)

    def to_json(self):
        return self.request

def main():
    query = MentionList()
    log = {}
    for fname in os.listdir(benchmarks_folder):
        path = os.path.join(benchmarks_folder, fname)
        for item in dataset_reader(path):
            query = MentionList()
            query.append_from_fields(**item)
            res = requests.post('http://localhost:5000/', json=query.to_json())
            log = {}
            try:
                print(res)
                log['fname'] = fname
                log['res'] = res.json()
            except Exception as e:
                print(e)
                log['fname'] = fname
                log['res'] = "error"
            with open(logfile, 'a+') as f:
                print(json.dumps(log), file=f)

if __name__ == '__main__':
    main()