import json,copy
from collections import defaultdict, OrderedDict
from nltk.tokenize import sent_tokenize, word_tokenize


def parse_decoding_results(filename, mode):
    predictions = json.load(open(filename))
    # test_file_with_idx = [lines.strip() for lines in open('utils/multiwoz.test.idx.txt')]
    test_file_with_idx = json.load(open(f'{mode}.idx.json'))
    res = defaultdict(list)
    res_bs = defaultdict(list)
    for prediction, file_idx in zip(predictions, test_file_with_idx):
        filename = file_idx#.split('@@@@@@@@@')[0].strip().split()[0]
        candidates = []
        candidates_bs = []
        belief_state = {}
        for lines in prediction:
            lines = lines.strip()
            if 'system :' in lines:
                system_response = lines.split('system :')[-1]
            else:
                system_response = ''
            system_response = ' '.join(word_tokenize(system_response))
            system_response = system_response.replace('[ ','[').replace(' ]',']')
            candidates.append(system_response)
            
        for lines in prediction:
            lines = lines.strip().split('system :')[0]            
            lines = ' '.join(lines.split()[:])
            domains = lines.split('|')
            belief_state = {}
            for domain_ in domains:
                if domain_ == '':
                    continue
                if len(domain_.split()) == 0:
                    continue
                domain = domain_.split()[0]
                if domain == 'none':
                    continue
                belief_state[domain] = {}
                svs = ' '.join(domain_.split()[1:]).split(';')
                for sv in svs:
                    if sv.strip() == '':
                        continue
                    sv = sv.split(' = ')
                    if len(sv) != 2:
                        continue
                    else:
                        s,v = sv
                    s = s.strip()
                    v = v.strip()
                    if v == "" or v == "dontcare" or v == 'not mentioned' or v == "don't care" or v == "dont care" or v == "do n't care" or v == 'none':
                        continue
                    belief_state[domain][s] = v
            candidates_bs.append(copy.copy(belief_state))

        def compare(key1, key2):
            key1 = key1[1]
            key2 = key2[1]
            if key1.count('[') > key2.count('['):
                return 1
            elif key1.count('[') == key2.count('['):
                return 1 if len(key1.split()) > len(key2.split()) else -1
            else:
                return -1

        import functools
        candidates_w_idx = [(idx, v) for idx,v in enumerate(candidates)]
        candidates = sorted(candidates_w_idx, key=functools.cmp_to_key(compare))
        if len(candidates) != 0:
            idx, value = candidates[-1]
            candidates_bs = candidates_bs[idx]
            candidates = value
        
        filename = filename.split('.')[0]
        res[filename].append(candidates)
        res_bs[filename].append(candidates_bs)

    response = OrderedDict(sorted(res.items()))
    belief = OrderedDict(sorted(res_bs.items()))

    return response, belief