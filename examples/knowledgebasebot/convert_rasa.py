import yaml,json
from collections import defaultdict
data = defaultdict(list)
import copy
import re
s = 'what [restaurants]{"entity": "object_type", "value": "restaurant"} can you recommend?"'


class NLUInstance:
    def __init__(self, string):
        self.state = []
        self.text = self.parse(string)
        self.state = ' ; '.join(self.state)
        print(self.text, self.state)
    def parse(self, string):
        texts = []
        states_strs = re.findall(r"\[(.+?)\}",string)
        for states_str in states_strs:
            text, state = states_str.split(']')
            state = eval(state + '}')
            if 'value' in state.keys():
                self.state.append(f'{state["entity"]} = {state["value"]}')
            else:
                self.state.append(f'{state["entity"]} = {text}')

        string = re.sub(r'\{(.+?)\}','',string)
        string = string.replace('[','').replace(']','')
        return string
        # for w in string.split():
        #     if '[' in w:
        #         texts.append(w[1:-1])
        #     else:
        #         texts.append(w)
        # return ' '.join(texts)    
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

for i in data['query_knowledge_base']:
    dialogue_turn = {}
    dialogue_turn['history'] = ['user : '+i.text]
    dialogue_turn['belief'] = 'belief : '+i.state
    dialogue_turn['kb'] = ''
    dialogue_turn['reply'] = 'system : action_query_knowledge_base'
    dialogue_example.append(copy.deepcopy(dialogue_turn))

for i in data['bot_challenge']:
    dialogue_turn = {}
    dialogue_turn['history'] = ['user : '+i.text]
    dialogue_turn['belief'] = 'belief : '+i.state
    dialogue_turn['kb'] = ''
    dialogue_turn['reply'] = 'system : bot_challenge'
    dialogue_example.append(copy.deepcopy(dialogue_turn))

for i in data['goodbye']:
    dialogue_turn = {}
    dialogue_turn['history'] = ['user : '+i.text]
    dialogue_turn['belief'] = 'belief : '+i.state
    dialogue_turn['kb'] = ''
    dialogue_turn['reply'] = 'system : good bye'
    dialogue_example.append(copy.deepcopy(dialogue_turn))

import random
random.shuffle(dialogue_example)
json.dump(dialogue_example, open('kb.soloist.json','w'), indent=2)