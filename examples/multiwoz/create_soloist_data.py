import json,copy
train_data = json.load(open('data/train.json'))
valid_data = json.load(open('data/valid.json'))
test_data = json.load(open('data/test.json'))

def write_data_for_internal_v1(data, output_file, output_idx_file):
    flist = []
    examples = []
    for fname, info in data.items():
        history = []
        example = {}
        for turn in info:
            user = turn['usr_no_delex'].strip().lower()
            system_nl = turn['sys'].strip().lower()
            system_no_delex_nl = turn['sys_no_delex'].strip().lower()
            kb = turn['db'].lower()
            ds = turn['bs']
            dp = 'dp : ' + turn['dp'][0]
            dp = dp.lower()
            active_domains = [i.split()[0] for i in ds]
            
            kb_nums = kb.split('|')[0].split(';')
            kb_nums_dict = dict([i.strip().split(' : ') for i in kb_nums])
            for active_domain in active_domains:
                if active_domain not in ['none','taxi','hospital']:
                    kb_nums = int(kb_nums_dict[active_domain])
                    # multiwoz style kb feature
                    if active_domain != 'train':
                        if kb_nums > 5:
                            kb_nums = 'more than five'
                        elif kb_nums == 0:
                            kb_nums = 'zero'
                        elif kb_nums == 1:
                            kb_nums = 'one'
                        elif kb_nums == 2:
                            kb_nums = 'two'
                        elif kb_nums == 3:
                            kb_nums = 'three'
                        elif kb_nums == 4:
                            kb_nums = 'four'
                    else:
                        if kb_nums > 40:
                            kb_nums = 'more than five'
                        elif kb_nums == 0:
                            kb_nums = 'zero'
                        elif kb_nums <= 2:
                            kb_nums = 'one'
                        elif kb_nums <= 5:
                            kb_nums = 'two'
                        elif kb_nums <= 10:
                            kb_nums = 'three'
                        elif kb_nums <= 40:
                            kb_nums = 'four'
                    kb = f'kb : {active_domain} {kb_nums}'
                else:
                    kb = f'kb : {active_domain}'
            
            history.append(f'user : {user}')

            user = f'user : {user} '
            if len(ds) == 0:
                ds = 'none'
            else:
                ds = ' | '.join(ds)
            ds = f'belief : {ds}'.lower()
            sys = f'system : {system_nl}'
            
            example['history'] = copy.copy(history)
            example['kb'] = kb
            example['belief'] = ds
            example['reply'] = sys
            example['name'] = fname
            example['dp'] = dp
            examples.append(copy.deepcopy(example))
            history.append(sys)
            flist.append(fname)

    json.dump(flist, open(output_idx_file,'w'))
    json.dump(examples, open(output_file,'w'),indent=2)
    print(len(examples))

write_data_for_internal_v1(train_data, 'train.soloist.json', 'train.idx.json')
write_data_for_internal_v1(valid_data, 'valid.soloist.json', 'valid.idx.json')
write_data_for_internal_v1(test_data, 'test.soloist.json', 'test.idx.json')