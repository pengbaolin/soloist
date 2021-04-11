import json
import copy
import glob
import sys
import random
examples = []
print(sys.argv)
for split in ['train','dev']:
    schema_info = json.load(open(f'dstc8-schema-guided-dialogue/{split}/schema.json'))
    schema_info = dict([(i['service_name'],i) for i in schema_info])
    for file in glob.glob(f'dstc8-schema-guided-dialogue/{split}/dialogues_*.json'):
        data = json.load(open(file))
        for dialogue in data:
            dialogue_id = dialogue['dialogue_id']
            services = dialogue['services'][0]
            schema = schema_info[services]
            description = schema['description']
            task_slots = [s['name'] for s in schema['slots']]
            task_intents = [s['name'] for s in schema['intents']]
            task_intents_description = [s['description'] for s in schema['intents']]
            turns = dialogue['turns']
            history = []
            example = {}
            for idx,turn in enumerate(turns):
                if idx == 0:
                    assert turn['speaker'] == 'USER'
                frame = turn['frames'][0]
                if turn['speaker'] == 'USER':
                    user_utter = turn['utterance']
                    history.append(f'user : {user_utter}')
                    belief_slot_values = frame['state']['slot_values']
                    slot_values_list = []
                    for slot_value in belief_slot_values.items():
                        slot,values = slot_value
                        value = values[0]
                        slot_values_list.append(f'{slot} = {value}')
                    slot_values_str = ' ; '.join(slot_values_list)

                else:
                    sys_utter = copy.copy(turn['utterance'])
                    
                    example['history'] = copy.copy(history)
                    
                    # slot_values_str = f'ds : {slot_values_str} '
                    slot_values_str = f'belief : {slot_values_str}'
                    example['belief'] = slot_values_str
                    slot_values_masked = []
                    if len(slot_values_list) != 0:
                        number_of_samples = random.choice([1,1,1])
                        if number_of_samples > len(slot_values_list):
                            number_of_samples = len(slot_values_list)
                        masked_idxs = random.sample(range(len(slot_values_list)), number_of_samples)
                        belief_masked_target = []
                        for masked_idx in sorted(masked_idxs):
                            belief_masked_target.append(slot_values_list[masked_idx])
                            slot_values_list[masked_idx] = '[MASK]'
                        example['belief_masked'] = 'belief : '+' ; '.join(slot_values_list)
                        example['belief_target'] = ' | '.join(belief_masked_target)
                    else:
                        example['belief_masked'] = 'none'
                        example['belief_target'] = 'none'
                    
                    slots = frame['slots']
                    offset = 0
                    len_ = len(sys_utter)
                    candidates = []
                    for idx,slot_info in enumerate(slots):
                        start, end, slot_name = slot_info['start'],slot_info['exclusive_end'],slot_info['slot']
                        sys_utter = sys_utter[:start+offset] + str(idx) * (end - start) + sys_utter[end+offset:]
                        candidates.append((slot_name, str(idx) * (end - start)))
                    for idx, info in enumerate(candidates):
                        slotname, target = info
                        sys_utter = sys_utter.replace(target, f'slot_{slotname}')
                    
                    reply = f'system : {sys_utter}'
                    example['reply'] = reply
                    example['original'] = turn['utterance']
                    example['dp'] = ''
                    examples.append(copy.deepcopy(example))
                    history.append(reply)


json.dump(examples, open('sgd.train.dev.json','w'), indent=2)