import yaml,json
from collections import defaultdict
data = defaultdict(list)
import copy

class NLUInstance:
    def __init__(self, string):
        self.state = []
        self.text = self.parse(string)
        self.state = ' ; '.join(self.state)
    
    def parse(self, string):
        texts = []
        for word in string.split():
            if '[' in word and '{' in word:
                tag = word.split('{')[1][:-1].split(':')[1][1:-1]
                text = word.split(']')[0][1:]
                texts.append(text)
                self.state.append(f'{tag} = {text}')
            else:
                texts.append(word)
        return ' '.join(texts)
    
    # def parse(self, string):
    #     texts = []
    #     for word in string.split():
    #         if '[' in word and '(' in word:
    #             tag = word.split('(')[1][:-1]
    #             text = word.split(']')[0][1:]
    #             texts.append(text)
    #             self.state.append(f'{tag} = {text}')
    #         else:
    #             texts.append(word)
    #     return ' '.join(texts)
    
    def __str__(self):
        return self.text + '\t' + ' ; '.join(self.state)

with open('rasa_nlu.yml') as f:
    nlu_data = yaml.load(f)

for nlu in nlu_data['nlu']:
    exmaples = nlu['examples'].split('\n')
    for exmaple in exmaples:
        exmaple = exmaple.replace('- ','')
        # print(NLUInstance(exmaple))
        data[nlu['intent']].append(NLUInstance(exmaple))

dialogue_example = []

for i in data['greet']:
    dialogue_turn = {}
    dialogue_turn['history'] = ['user : '+i.text]
    dialogue_turn['belief'] = 'belief : '+i.state
    dialogue_turn['kb'] = ''
    dialogue_turn['reply'] = 'system : What can I do for you?'
    dialogue_example.append(copy.deepcopy(dialogue_turn))

for i in data['ask_remind_call']:
    dialogue_turn = {}
    dialogue_turn['history'] = ['user : '+i.text]
    dialogue_turn['belief'] = 'belief : '+i.state
    dialogue_turn['kb'] = ''
    dialogue_turn['reply'] = 'system : action_set_reminder'
    dialogue_example.append(copy.deepcopy(dialogue_turn))

for i in data['ask_forget_reminders']:
    dialogue_turn = {}
    dialogue_turn['history'] = ['user : '+i.text]
    dialogue_turn['belief'] = 'belief : '+i.state
    dialogue_turn['kb'] = ''
    dialogue_turn['reply'] = 'system : action_forget_reminders'
    dialogue_example.append(copy.deepcopy(dialogue_turn))

for i in data['bye']:
    dialogue_turn = {}
    dialogue_turn['history'] = ['user : '+i.text]
    dialogue_turn['belief'] = 'belief : '+i.state
    dialogue_turn['kb'] = ''
    dialogue_turn['reply'] = 'system : Bye'
    dialogue_example.append(copy.deepcopy(dialogue_turn))

import random
random.shuffle(dialogue_example)
json.dump(dialogue_example, open('reminderbot.soloist.json','w'), indent=2)