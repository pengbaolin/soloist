from flask import Flask, request, Response, jsonify
from flask import render_template
from flask_cors import CORS
import flask
import json
from collections import defaultdict
import random

import sys
sys.path.append('../../')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

app = Flask(__name__)
CORS(app)

from queue import Queue
from threading import Thread

rgi_queue = Queue(maxsize=0)
rgo_queue = Queue(maxsize=0)

def parse(sampled_results):
    print('----------------------')
    print(sampled_results)
    candidates = []
    for system_response in sampled_results:
        system_response = system_response.split('system :')[-1]
        system_response = ' '.join(word_tokenize(system_response))
        system_response = system_response.replace('[ ','[').replace(' ]',']')
        candidates.append(system_response)

    candidates_bs = []
    for system_response in sampled_results:
        system_response = system_response.strip()
        system_response = system_response.split('system :')[0]
        system_response = ' '.join(system_response.split()[:])
        svs = system_response.split(' ; ')
        bs_state = {}
        for sv in svs:
            if '=' in sv:
                s,v = sv.split('=')
                s = s.strip()
                v = v.strip()
                bs_state[s] = v
        candidates_bs.append(copy.copy(bs_state))

    candidates_w_idx = [(idx, v) for idx,v in enumerate(candidates)]
    candidates = sorted(candidates_w_idx, key=functools.cmp_to_key(compare))

    idx, response = candidates[-1]
    states = candidates_bs[idx]
    return states,response

def compare(key1, key2):
    key1 = key1[1]
    key2 = key2[1]
    if key1.count('[') > key2.count('['):
        return 1
    elif key1.count('[') == key2.count('['):
        return 1 if len(key1.split()) > len(key2.split()) else -1
    else:
        return -1  

def predictor(context):
    context_formated = []
    for idx, i in enumerate(context):
        if idx % 2 == 0:
            context_formated.append(f'user : {i}')
        else:
            context_formated.append(f'system : {i}')

    sampled_results = sample(context_formated[-1:])
    belief_states, response = parse(sampled_results)

    return response, belief_states

global_counter = 0
@app.route('/generate', methods=['GET','POST'])
def generate_queue():
    global global_counter, rgi_queue, rgo_queue
    try:
        in_request = request.json
        print(in_request)
    except:
        return "invalid input: "
    global_counter += 1
    rgi_queue.put((global_counter, in_request))
    output = rgo_queue.get()
    rgo_queue.task_done()
    return jsonify(output)

def generate_for_queue(in_queue, out_queue):
    memory = []
    while True:
        _, in_request = in_queue.get()
        obs = in_request['msg']
        response, belief_states = predictor(obs)    
        if belief_states != {}:
            name = belief_states['name']
            memory.append(f'reminder call [name]({name})')
        
        followup = ''
        if response.strip() == 'action_set_reminder':
            followup = 'Sure thing, added to your reminder list!'
        if response.strip() == 'action_forget_reminders':
            followup = 'Sure thing, remove all your reminders!'
            memory = []
        res = {}
        res['response'] = response
        res['memory'] = memory
        res['followup'] = followup
        out_queue.put(res)
        in_queue.task_done()

if __name__ == "__main__":

    from soloist.server import *
    args.model_name_or_path = 'reminderbot_model'
    main()
    worker = Thread(target=generate_for_queue, args=(rgi_queue, rgo_queue,))
    worker.setDaemon(True)
    worker.start()
    app.run(host='0.0.0.0',port=8081)