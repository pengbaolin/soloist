import math
import utils.delexicalize as delex
from collections import Counter
from nltk.util import ngrams
import json
from utils.nlp import normalize
import sqlite3
import os
import random
import logging
from utils.nlp import BLEUScorer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, sentence_bleu

file=open

class BaseEvaluator(object):
    def initialize(self):
        raise NotImplementedError

    def add_example(self, ref, hyp):
        raise NotImplementedError

    def get_report(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _get_prec_recall(tp, fp, fn):
        precision = tp / (tp + fp + 10e-20)
        recall = tp / (tp + fn + 10e-20)
        f1 = 2 * precision * recall / (precision + recall + 1e-20)
        return precision, recall, f1

    @staticmethod
    def _get_tp_fp_fn(label_list, pred_list):
        tp = len([t for t in pred_list if t in label_list])
        fp = max(0, len(pred_list) - tp)
        fn = max(0, len(label_list) - tp)
        return tp, fp, fn


class BLEUScorer(object):
    ## BLEU score calculator via GentScorer interface
    ## it calculates the BLEU-4 by taking the entire corpus in
    ## Calulate based multiple candidates against multiple references
    def score(self, hypothesis, corpus, n=1):
        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]

        # accumulate ngram statistics
        for hyps, refs in zip(hypothesis, corpus):
            # if type(hyps[0]) is list:
            #    hyps = [hyp.split() for hyp in hyps[0]]
            # else:
            #    hyps = [hyp.split() for hyp in hyps]
            # import pdb
            # pdb.set_trace()
            hyps = [hyp.split() for hyp in hyps]
            refs = [ref.split() for ref in refs]
            # import pdb
            # pdb.set_trace()
            # hyps = [hyps]
            # Shawn's evaluation
            # refs[0] = [u'GO_'] + refs[0] + [u'EOS_']
            # hyps[0] = [u'GO_'] + hyps[0] + [u'EOS_']

            for idx, hyp in enumerate(hyps):
                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt
                    # import pdb
                    # pdb.set_trace()
                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng])) \
                                   for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0: break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)
                if n == 1:
                    break
        # computing bleu score
        p0 = 1e-7
        bp = 1 if c > r else math.exp(1 - float(r) / float(c))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 \
                for i in range(4)]
        s = math.fsum(w * math.log(p_n) \
                      for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        return bleu


class MultiWozDB(object):
    # loading databases
    domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital']  # , 'police']
    dbs = {}
    CUR_DIR = os.path.dirname(__file__)

    for domain in domains:
        db = os.path.join('db/{}-dbase.db'.format(domain))
        conn = sqlite3.connect(db)
        c = conn.cursor()
        dbs[domain] = c

    def queryResultVenues(self, domain, turn, bs=None, real_belief=False):
        # query the db
        sql_query = "select * from {}".format(domain)
        # import pdb
        # pdb.set_trace()
        if real_belief == True:
            items = turn.items()
        else:
            items = turn['metadata'][domain]['semi'].items()

        # if bs is None:
            # return []

        if bs is not None:
            items = bs.items()
        #     # print(bs, turn.items())
            if len(items) == 0:
                return []
            # import pdb
            # pdb.set_trace()
        # else:
            # items = []
            # if bs['domain'] == domain:
            #     items = bs.items()
            #     bs['domain'] = ''
            # else:
                # items = []
            # items_ = bs.items()
            # items_remains = {}
            # items_all = dict(items)

            # for k, v in items_:
            #     # try:
            #         # items_remains[k] = items_all[k]
            #     # except Exception:
            #         # continue
                
            #     if k in items_all.keys():
            #         # items_remains[k] = items_all[k]
            #         items_remains[k] = v
            # items = items_remains.items()
            # # print(items)
            # items = items_
            # import pdb

            
        flag = True
        for key, val in items:
            if val == "" or val == "dontcare" or val == 'not mentioned' or val == "don't care" or val == "dont care" or val == "do n't care" or val == "none":
                pass
            else:
                if flag:
                    sql_query += " where "
                    val2 = val.replace("'", "''")
                    val2 = normalize(val2)
                    if key.lower() == 'leaveAt'.lower():
                        sql_query += r" " + key + " > " + r"'" + val2 + r"'"
                    elif key.lower() == 'arriveBy'.lower():
                        sql_query += r" " + key + " < " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                    flag = False
                else:
                    val2 = val.replace("'", "''")
                    val2 = normalize(val2)
                    if key.lower() == 'leaveAt'.lower():
                        sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                    elif key.lower() == 'arriveBy'.lower():
                        sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                    else:
                        sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

        try:  # "select * from attraction  where name = 'queens college'"
            # print(sql_query)
            return self.dbs[domain].execute(sql_query).fetchall()
        except:
            return []  # TODO test it


class MultiWozEvaluator(BaseEvaluator):
    def __init__(self, data_name):
        self.data_name = data_name
        self.slot_dict = delex.prepareSlotValuesIndependent()
        self.delex_dialogues = json.load(file('data/multi-woz/delex.json'))
        # self.delex_dialogues = json.load(file('/home/bapeng/experiment/multiwoz2.1/MultiWOZ_2.1/data.json'))
        
        self.db = MultiWozDB()
        self.labels = list()
        self.hyps = list()

    def add_example(self, ref, hyp):
        self.labels.append(ref)
        self.hyps.append(hyp)

    def _parseGoal(self, goal, d, domain):
        """Parses user goal into dictionary format."""
        goal[domain] = {}
        goal[domain] = {'informable': [], 'requestable': [], 'booking': []}
        if 'info' in d['goal'][domain]:
        # if d['goal'][domain].has_key('info'):
            if domain == 'train':
                # we consider dialogues only where train had to be booked!
                if 'book' in d['goal'][domain]:
                # if d['goal'][domain].has_key('book'):
                    goal[domain]['requestable'].append('reference')
                if 'reqt' in d['goal'][domain]:
                # if d['goal'][domain].has_key('reqt'):
                    if 'trainID' in d['goal'][domain]['reqt']:
                        goal[domain]['requestable'].append('id')
            else:
                if 'reqt' in d['goal'][domain]:
                # if d['goal'][domain].has_key('reqt'):
                    for s in d['goal'][domain]['reqt']:  # addtional requests:
                        if s in ['phone', 'address', 'postcode', 'reference', 'id']:
                            # ones that can be easily delexicalized
                            goal[domain]['requestable'].append(s)
                if 'book' in d['goal'][domain]:
                # if d['goal'][domain].has_key('book'):
                    goal[domain]['requestable'].append("reference")

            goal[domain]["informable"] = d['goal'][domain]['info']
            if 'book' in d['goal'][domain]:
            # if d['goal'][domain].has_key('book'):
                goal[domain]["booking"] = d['goal'][domain]['book']

        return goal

    def _evaluateGeneratedDialogue(self, dialog, goal, realDialogue, real_requestables, soft_acc=False):
        """Evaluates the dialogue created by the model.
        First we load the user goal of the dialogue, then for each turn
        generated by the system we look for key-words.
        For the Inform rate we look whether the entity was proposed.
        For the Success rate we look for requestables slots"""
        # for computing corpus success
        requestables = ['phone', 'address', 'postcode', 'reference', 'id']

        dialog, bs = dialog
        # import pdb
        # pdb.set_trace()
        # CHECK IF MATCH HAPPENED
        provided_requestables = {}
        venue_offered = {}
        domains_in_goal = []

        for domain in goal.keys():
            venue_offered[domain] = []
            provided_requestables[domain] = []
            domains_in_goal.append(domain)

        # state = {}
        # state['restaurant'] = {}
        # state['hotel'] = {}
        # state['attraction'] = {}
        # state['train'] = {}
        for t, sent_t in enumerate(dialog):
            for domain in goal.keys():
                # for computing success
                if '[' + domain + '_name]' in sent_t or '_id' in sent_t:
                    if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                        # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION
                        # import pdb
                        # pdb.set_trace()
                        # if domain != bs[t]['domain']:
                            # print('hit')
                            # continue
                        if domain in bs[t].keys():
                            state = bs[t][domain]
                        else:
                            state = {}
                        # if domain in bs[t].keys():
                            # state[domain].update(bs[t][domain])
                        # else:
                            # state = {}


                        venues = self.db.queryResultVenues(domain, realDialogue['log'][t * 2 + 1], state, real_belief=False)

                        # if venue has changed
                        if len(venue_offered[domain]) == 0 and venues:
                            venue_offered[domain] = random.sample(venues, 1)
                        else:
                            flag = False
                            for ven in venues:
                                if venue_offered[domain][0] == ven:
                                    flag = True
                                    break
                            if not flag and venues:  # sometimes there are no results so sample won't work
                                # print venues
                                venue_offered[domain] = random.sample(venues, 1)
                    else:  # not limited so we can provide one
                        venue_offered[domain] = '[' + domain + '_name]'

                # ATTENTION: assumption here - we didn't provide phone or address twice! etc
                for requestable in requestables:
                    if requestable == 'reference':
                        if domain + '_reference' in sent_t:
                            if 'restaurant_reference' in sent_t:
                                if realDialogue['log'][t * 2]['db_pointer'][
                                    -5] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            elif 'hotel_reference' in sent_t:
                                if realDialogue['log'][t * 2]['db_pointer'][
                                    -3] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            elif 'train_reference' in sent_t:
                                if realDialogue['log'][t * 2]['db_pointer'][
                                    -1] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            else:
                                provided_requestables[domain].append('reference')
                    else:
                        if domain + '_' + requestable + ']' in sent_t:
                            provided_requestables[domain].append(requestable)

        # if name was given in the task
        for domain in goal.keys():
            # if name was provided for the user, the match is being done automatically
            # if realDialogue['goal'][domain].has_key('info'):
            if 'info' in realDialogue['goal'][domain]:
                # if realDialogue['goal'][domain]['info'].has_key('name'):
                if 'name' in realDialogue['goal'][domain]['info']:
                    venue_offered[domain] = '[' + domain + '_name]'

            # special domains - entity does not need to be provided
            if domain in ['taxi', 'police', 'hospital']:
                venue_offered[domain] = '[' + domain + '_name]'

            # the original method
            # if domain == 'train':
            #     if not venue_offered[domain]:
            #         # if realDialogue['goal'][domain].has_key('reqt') and 'id' not in realDialogue['goal'][domain]['reqt']:
            #         if 'reqt' in realDialogue['goal'][domain] and 'id' not in realDialogue['goal'][domain]['reqt']:
            #             venue_offered[domain] = '[' + domain + '_name]'

            # Wrong one in HDSA
            # if domain == 'train':
            #     if not venue_offered[domain]:
            #         if goal[domain]['requestable'] and 'id' not in goal[domain]['requestable']:
            #             venue_offered[domain] = '[' + domain + '_name]'

            # if id was not requested but train was found we dont want to override it to check if we booked the right train
            if domain == 'train' and (not venue_offered[domain] and 'id' not in goal['train']['requestable']):
                venue_offered[domain] = '[' + domain + '_name]'

        """
        Given all inform and requestable slots
        we go through each domain from the user goal
        and check whether right entity was provided and
        all requestable slots were given to the user.
        The dialogue is successful if that's the case for all domains.
        """
        # HARD EVAL
        stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                 'taxi': [0, 0, 0],
                 'hospital': [0, 0, 0], 'police': [0, 0, 0]}

        match = 0
        success = 0
        # MATCH
        for domain in goal.keys():
            match_stat = 0
            if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                goal_venues = self.db.queryResultVenues(domain, goal[domain]['informable'], real_belief=True)
                if type(venue_offered[domain]) is str and '_name' in venue_offered[domain]:
                    match += 1
                    match_stat = 1
                elif len(venue_offered[domain]) > 0 and venue_offered[domain][0] in goal_venues:
                    match += 1
                    match_stat = 1
            else:
                if domain + '_name]' in venue_offered[domain]:
                    match += 1
                    match_stat = 1

            stats[domain][0] = match_stat
            stats[domain][2] = 1

        if soft_acc:
            match = float(match)/len(goal.keys())
        else:
            if match == len(goal.keys()):
                match = 1.0
            else:
                match = 0.0

        # SUCCESS
        if match == 1.0:
            for domain in domains_in_goal:
                success_stat = 0
                domain_success = 0
                if len(real_requestables[domain]) == 0:
                    success += 1
                    success_stat = 1
                    stats[domain][1] = success_stat
                    continue
                # if values in sentences are super set of requestables
                for request in set(provided_requestables[domain]):
                    if request in real_requestables[domain]:
                        domain_success += 1

                if domain_success >= len(real_requestables[domain]):
                    success += 1
                    success_stat = 1

                stats[domain][1] = success_stat

            # final eval
            if soft_acc:
                success = float(success)/len(real_requestables)
            else:
                if success >= len(real_requestables):
                    success = 1
                else:
                    success = 0

        # rint requests, 'DIFF', requests_real, 'SUCC', success
        return success, match, stats

    def _evaluateRealDialogue(self, dialog, filename):
        """Evaluation of the real dialogue.
        First we loads the user goal and then go through the dialogue history.
        Similar to evaluateGeneratedDialogue above."""
        domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
        requestables = ['phone', 'address', 'postcode', 'reference', 'id']

        # get the list of domains in the goal
        domains_in_goal = []
        goal = {}
        for domain in domains:
            if dialog['goal'][domain]:
                goal = self._parseGoal(goal, dialog, domain)
                domains_in_goal.append(domain)

        # compute corpus success
        real_requestables = {}
        provided_requestables = {}
        venue_offered = {}
        for domain in goal.keys():
            provided_requestables[domain] = []
            venue_offered[domain] = []
            real_requestables[domain] = goal[domain]['requestable']

        # iterate each turn
        m_targetutt = [turn['text'] for idx, turn in enumerate(dialog['log']) if idx % 2 == 1]
        for t in range(len(m_targetutt)):
            for domain in domains_in_goal:
                sent_t = m_targetutt[t]
                # for computing match - where there are limited entities
                if domain + '_name' in sent_t or '_id' in sent_t:
                    if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                        # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION
                        venues = self.db.queryResultVenues(domain, dialog['log'][t * 2 + 1])

                        # if venue has changed
                        if len(venue_offered[domain]) == 0 and venues:
                            venue_offered[domain] = random.sample(venues, 1)
                        else:
                            flag = False
                            for ven in venues:
                                if venue_offered[domain][0] == ven:
                                    flag = True
                                    break
                            if not flag and venues:  # sometimes there are no results so sample won't work
                                # print venues
                                venue_offered[domain] = random.sample(venues, 1)
                    else:  # not limited so we can provide one
                        venue_offered[domain] = '[' + domain + '_name]'

                for requestable in requestables:
                    # check if reference could be issued
                    if requestable == 'reference':
                        if domain + '_reference' in sent_t:
                            if 'restaurant_reference' in sent_t:
                                if dialog['log'][t * 2]['db_pointer'][-5] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            elif 'hotel_reference' in sent_t:
                                if dialog['log'][t * 2]['db_pointer'][-3] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                                    # return goal, 0, match, real_requestables
                            elif 'train_reference' in sent_t:
                                if dialog['log'][t * 2]['db_pointer'][-1] == 1:  # if pointer was allowing for that?
                                    provided_requestables[domain].append('reference')

                            else:
                                provided_requestables[domain].append('reference')
                    else:
                        if domain + '_' + requestable in sent_t:
                            provided_requestables[domain].append(requestable)

        # offer was made?
        for domain in domains_in_goal:
            # if name was provided for the user, the match is being done automatically
            # if dialog['goal'][domain].has_key('info'):
            if 'info' in dialog['goal'][domain]:
                # if dialog['goal'][domain]['info'].has_key('name'):
                if 'name' in dialog['goal'][domain]['info']:
                    venue_offered[domain] = '[' + domain + '_name]'

            # special domains - entity does not need to be provided
            if domain in ['taxi', 'police', 'hospital']:
                venue_offered[domain] = '[' + domain + '_name]'

            # if id was not requested but train was found we dont want to override it to check if we booked the right train
            if domain == 'train' and (not venue_offered[domain] and 'id' not in goal['train']['requestable']):
                venue_offered[domain] = '[' + domain + '_name]'

        # HARD (0-1) EVAL
        stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                 'taxi': [0, 0, 0],
                 'hospital': [0, 0, 0], 'police': [0, 0, 0]}

        match, success = 0, 0
        # MATCH
        for domain in goal.keys():
            match_stat = 0
            if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                goal_venues = self.db.queryResultVenues(domain, dialog['goal'][domain]['info'], real_belief=True)
                # print(goal_venues)
                if type(venue_offered[domain]) is str and '_name' in venue_offered[domain]:
                    match += 1
                    match_stat = 1
                elif len(venue_offered[domain]) > 0 and venue_offered[domain][0] in goal_venues:
                    match += 1
                    match_stat = 1

            else:
                if domain + '_name' in venue_offered[domain]:
                    match += 1
                    match_stat = 1

            stats[domain][0] = match_stat
            stats[domain][2] = 1

        if match == len(goal.keys()):
            match = 1
        else:
            match = 0

        # SUCCESS
        if match:
            for domain in domains_in_goal:
                domain_success = 0
                success_stat = 0
                if len(real_requestables[domain]) == 0:
                    # check that
                    success += 1
                    success_stat = 1
                    stats[domain][1] = success_stat
                    continue
                # if values in sentences are super set of requestables
                for request in set(provided_requestables[domain]):
                    if request in real_requestables[domain]:
                        domain_success += 1

                if domain_success >= len(real_requestables[domain]):
                    success += 1
                    success_stat = 1

                stats[domain][1] = success_stat

            # final eval
            if success >= len(real_requestables):
                success = 1
            else:
                success = 0

        return goal, success, match, real_requestables, stats

    def _parse_entities(self, tokens):
        entities = []
        for t in tokens:
            if '[' in t and ']' in t:
                entities.append(t)
        return entities
    def evaluteDST(self, dialogues, dialogues_bs, real_dialogues=False, mode='valid'):
        delex_dialogues = self.delex_dialogues
        match = 0
        total = 0
        for filename, dial in dialogues.items():
            
            filename = filename.upper().split('.')[0]
            filename = filename+'.json'
            # try:
            data = delex_dialogues[filename]
            bs_state = dialogues_bs[filename.split('.')[0]]

            for t, sent_t in enumerate(dial):
                turn_dict_ = {}
                predict_turn_dst = bs_state[t]
                # domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi']
                domains = data['log'][t * 2 + 1]['metadata'].keys()
                # for domain in data['log'][t * 2 + 1]['metadata'].keys():
                for domain in domains:
                    
                    true_turn_dst = data['log'][t * 2 + 1]['metadata'][domain]['semi'].items()
                    if len(true_turn_dst) <= 0:
                        continue
                    turn_dict_[domain] = {}
                    for k,val in true_turn_dst:
                        # if k == 'name':
                            # continue
                        if val == "" or val == "dontcare" or val == 'not mentioned' or val == "don't care" or val == "dont care" or val == "do n't care" or val == 'none':
                            continue
                        else:
                            turn_dict_[domain][k.lower()] = val.lower()
                    true_turn_dst_booking = data['log'][t * 2 + 1]['metadata'][domain]['book'].items()
                    if len(true_turn_dst_booking) > 1 and not 'booking' in turn_dict_.keys():
                        turn_dict_['booking'] = {}
                    for k,val in true_turn_dst_booking:
                        if k == 'booked':
                            continue
                        if val == '' or val == "dontcare" or val == 'not mentioned' or val == "don't care" or val == "dont care" or val == "do n't care" or val == 'none':
                            continue
                        else:
                            turn_dict_['booking'][k.lower()] = val.lower()
                redundent_k = []
                for k,v in turn_dict_.items():
                    if len(v) == 0:
                        redundent_k.append(k)
                for k in redundent_k:
                    del turn_dict_[k]
                # if 'booking' in predict_turn_dst.keys():
                #     candidate = []
                #     domains = predict_turn_dst.keys()
                #     for domain in domains:
                #         if 'booking' == domain:
                #             continue
                #         if 'name' in predict_turn_dst[domain].keys():
                #             candidate.append(domain)
                #     for c in candidate:
                #         del predict_turn_dst[c]['name']
                
                # reducent_domain = []
                # for domain in predict_turn_dst.keys():
                #     if not domain in domains:
                #         # del predict_turn_dst[domain]
                #         reducent_domain.append(domain)
                # # # reducent_domain.append('booking')
                # for d in reducent_domain:
                #     del predict_turn_dst[d]
                # import pdb
                # pdb.set_trace()
                match += turn_dict_ == predict_turn_dst
                    # else:
                #     if 'name' in  predict_turn_dst[domain]:
                #         del predict_turn_dst[domain]['name']
                # if turn_dict_ != predict_turn_dst:
                    # print(turn_dict_)
                    # print(predict_turn_dst)
                    # print('-'*100)
                    # import pdb
                    # pdb.set_trace()
                total += 1
            # except Exception:
            #     print(Exception)
            #     continue
        print('-'*100)
        print(f'DST:\t{match/total, match, total}')
        print('-'*100)

    def evaluateModel(self, dialogues, dialogues_bs, real_dialogues=False, mode='valid'):
        """Gathers statistics for the whole sets."""
        delex_dialogues = self.delex_dialogues
        successes, matches = 0, 0
        total = 0

        gen_stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                     'taxi': [0, 0, 0],
                     'hospital': [0, 0, 0], 'police': [0, 0, 0]}
        sng_gen_stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                         'taxi': [0, 0, 0], 'hospital': [0, 0, 0], 'police': [0, 0, 0]}
        all_slots = 0
        hit_slots = 0
        for filename, dial in dialogues.items():
            
            filename = filename.upper().split('.')[0]
            filename = filename+'.json'
            # import pdb
            # pdb.set_trace()
            try:
                
                data = delex_dialogues[filename]
                bs_state = dialogues_bs[filename.split('.')[0]]
            except Exception:
                print(filename)
                continue
            # assert len(data['log']) == len(bs_state)
            
            # for _i in range(len(data['log'])//2):
            #     # real_belief_str = []
                
            #     for _domain,_states in data['log'][_i*2+1]['metadata'].items():
            #         # vstr = []
            #         # slot_list = []
            #         for _s,_v in _states['semi'].items():
            #             if _v != 'not mentioned' and _v != '':
            #                 # vstr.append(f'{_s} = {_v}')
            #                 # slot_list.append(_s)
            #                 # import pdb
            #                 # pdb.set_trace()
            #                 all_slots += 1
            #                 if _domain not in bs_state[_i].keys():
            #                     # import pdb
            #                     # pdb.set_trace()
            #                     continue
            #                 else:
            #                     if _s == 'leaveAt':
            #                         _s = 'leaveat'
            #                     if _s == 'arriveBy':
            #                         _s = 'arriveby'
            #                     if _s not in bs_state[_i][_domain].keys():
            #                         # print(_s, bs_state[_i][_domain].keys())
            #                         continue
            #                     else:
            #                         hit_slots +=  _v == bs_state[_i][_domain][_s]
                #     vstr = ' '.join(sorted(vstr))
                # real_belief_str.append(f'{_domain} {vstr}')
                # real_belief_str = ' '.join(sorted(real_belief_str))
                # predicted_belief_str = []
                # for _domain in bs_state[_i].keys():
                #     if _domain == 'booking':
                #         continue
                #     vstr = []
                #     # print(bs_state[_i][_domain].items())
                #     for _s,_v in bs_state[_i][_domain].items():
                #         if _s == 'leaveat':
                #             _s = 'leaveAt'
                #         if _s == 'arriveby':
                #             _s = 'arriveBy'
                #         if _s in slot_list:
                #             vstr.append(f'{_s} = {_v}')
                #     vstr = ' '.join(sorted(vstr))
                #     predicted_belief_str.append(f'{_domain} {vstr}')
                # predicted_belief_str = ' '.join(sorted(predicted_belief_str))
                # print(predicted_belief_str, real_belief_str)
                # if real_belief_str.count(' ') == 1 and predicted_belief_str.count(' ') == 1:
                #     hit_slots += 1
                # else:    
                #     hit_slots += predicted_belief_str == real_belief_str
                # # if not predicted_belief_str == real_belief_str:
                #     # import pdb
                #     # pdb.set_trace()
                # all_slots += 1
            # print(hit_slots, all_slots)

            goal, success, match, requestables, _ = self._evaluateRealDialogue(data, filename)
            success, match, stats = self._evaluateGeneratedDialogue((dial,bs_state), goal, data, requestables,
                                                                    soft_acc=mode =='soft')

            successes += success
            matches += match
            total += 1

            for domain in gen_stats.keys():
                gen_stats[domain][0] += stats[domain][0]
                gen_stats[domain][1] += stats[domain][1]
                gen_stats[domain][2] += stats[domain][2]

            if 'SNG' in filename:
                for domain in gen_stats.keys():
                    sng_gen_stats[domain][0] += stats[domain][0]
                    sng_gen_stats[domain][1] += stats[domain][1]
                    sng_gen_stats[domain][2] += stats[domain][2]
        print(hit_slots, all_slots)
        if real_dialogues:
            # BLUE SCORE
            corpus = []
            model_corpus = []
            bscorer = BLEUScorer()

            for dialogue in dialogues.keys():
                # data = delex_dialogues[dialogue]
                data = delex_dialogues[dialogue.split('.')[0].upper()+'.json']
                model_turns, corpus_turns = [], []
                sys_turns = data['log']
                for idx, turn in enumerate(sys_turns):
                    if idx % 2 == 1:
                        corpus_turns.append([turn['text']])
                for turn in dialogues[dialogue]:
                    model_turns.append([turn])

                if len(model_turns) == len(corpus_turns):
                    corpus.extend(corpus_turns)
                    model_corpus.extend(model_turns)
                else:
                    raise('Wrong amount of turns')
            blue_score = bscorer.score(model_corpus, corpus)
            smooth = SmoothingFunction()
            corpus_ = [[i[0]] for i in corpus]
            hypothesis_ = [i[0] for i in model_corpus]
            # for i_, j_ in zip(corpus_, hypothesis_):
                # print(i_,j_, sentence_bleu(i_,j_))
            print('corpus level', corpus_bleu(corpus_, hypothesis_, smoothing_function=smooth.method1))
        else:
            blue_score = 0.

        report = ""
        report += '{} Corpus Matches : {:2.2f}%'.format(mode, (matches / float(total) * 100)) + "\n"
        report += '{} Corpus Success : {:2.2f}%'.format(mode, (successes / float(total) * 100)) + "\n"
        report += '{} Corpus BLEU : {:2.2f}%'.format(mode, blue_score) + "\n"
        report += 'Total number of dialogues: %s ' % total

        print(report)
        combined = (successes/float(total) + matches/float(total))/2 + blue_score
        print(f'Combined Score {combined}')

        return report, successes/float(total), matches/float(total)

from parser import parse_decoding_results
import glob
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", default=None, type=str, required=True, help="The input evaluation file.")
    parser.add_argument("--eval_mode", default='test', type=str, help="valid/test")

    args = parser.parse_args()
    mode = "test"
    evaluator = MultiWozEvaluator(mode)
        
    f = 'soloist_internal_v5_hist15_special_b4_nc1_mc0.33_bsp_bp_rp.checkpoint-75000.ns5.k0.p0.5.t1.aug28.v5.txt'
    res, res_bs = parse_decoding_results(f, mode)
    human_proc_data = res

    # PROVIDE HERE YOUR GENERATED DIALOGUES INSTEAD
    generated_data = human_proc_data
    generated_proc_belief = res_bs
    evaluator.evaluteDST(generated_data, generated_proc_belief, True, mode=mode)
    evaluator.evaluateModel(generated_data, generated_proc_belief, True, mode=mode)
    
        
