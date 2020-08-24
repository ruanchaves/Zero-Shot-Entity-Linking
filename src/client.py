import requests, json
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
nlp = English()
tokenizer = Tokenizer(nlp.vocab)

class MentionList(object):

    def __init__(self):
        self.nlp = English()
        self.tokenizer = self.nlp.Defaults.create_tokenizer(nlp)
        self.sep_left = "[unused1]"
        self.sep_right = "[unused2]"
        self.token_gold_dui = "639E07B4F4A90B2C"
        self.gold_world = "yugioh"
        self.counter = 0
        self.request = {}
        self.friendly_sep_left = "["
        self.friendly_sep_right = "]"

    def _tokenize(self, text):
        return [doc.text for doc in self.tokenizer(text)]
    
    def append(self, text):

        friendly_sep_left_position = text.find(self.friendly_sep_left)
        friendly_sep_right_position = text.find(self.friendly_sep_right)

        context_left = text[0:friendly_sep_left_position-1]
        mention = text[friendly_sep_left_position+1:friendly_sep_right_position]
        context_right = text[friendly_sep_right_position+2:len(text)]

        if not mention:
            raise NotImplementedError

        cl = self._tokenize(context_left)
        cr = self._tokenize(context_right)
        m = self._tokenize(mention)
        anchored_context = cl + [self.sep_left] + m + [self.sep_right] + cr

        self.request[str(self.counter)] = {
            "raw_mention": mention,
            "gold_dui": self.token_gold_dui,
            "gold_world": self.gold_world,
            "anchored_context": anchored_context, 
        }

        self.counter += 1

    def extend(self, lst):
        for item in lst:
            self.append(item)

    def __repr__(self):
        return json.dumps(self.request, indent=4)

    def to_json(self):
        return self.request

def main():
    mc = MentionList()
    text = "He sinks to his knees, a hand covering his forehead, and as Mokuba asks if Kaiba's okay, Kaiba asks why the Gods showed him [this] horrifying knowledge."
    mc.append(text)
    print(mc)
    res = requests.post('http://localhost:5000/', json=mc.to_json())
    if res.ok:
        print(res.json())
    else:
        print(res)

if __name__ == '__main__':
    main()