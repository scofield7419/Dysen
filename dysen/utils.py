import time
import os
import json
import math
import random
import re
import numpy as np
import pandas as pd
import errno
import torch


def load_json(path):
    with open(path, 'r') as f:
        x = json.load(f)
    return x


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def load_data(args):
    cand_data = load_json(os.path.join(args.sampled_data_dir, f'train_candidate_{args.cand_number}.json'))
    cand_ids, cand_examples = cand_data['id'], cand_data['data']

    # cand_examples = data_raw_train
    # cand_ids = list(range(len(cand_examples)))

    all_train_feats = torch.load(args.feature_path)

    return cand_examples, cand_ids, all_train_feats


class Logger(object):

    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.log_file = open(output_name, 'w')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        self.log_file.write(str(msg) + '\n')
        self.log_file.flush()
        print(msg)


def extract_prediction_for_action_planning(input_string):
    # Output:
    # (woman, putting, bottle of water, (0v, 1v)),
    # (woman, taking, fitness mat, (1v, 2v)),
    # (woman, get out, office, (2v, 3v)).

    # Regular expression pattern to match event triplets
    pattern = r'(\b(\w+\s*\w*)\s*, \b(\w+\s*\w*)\s*, \b(\w+\s*\w*)\s*, (\d+v, \d+v))'
    # pattern = r'\b(\w+\s*\w*)\s*:\s*\[(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)\]'

    # Find all matches of the pattern in the input string
    matches = re.findall(pattern, input_string)

    # Extract categories and bounding boxes from the matches
    event_triplets = []
    durations = []
    # bboxes = []
    for match in matches:
        event_triplets.append((match[0], match[1], match[2]))
        durations.append((int(match[3]), int(match[4])))
        # bboxes.append([float(match[1]), float(match[2]), float(match[3]), float(match[4])])

    # Return the extracted event_triplets and durations
    return event_triplets, durations


def build_dsg_from_action_planning(action_list, duration_list):
    dsg = []
    sg_num = duration_list[-1][-1]+1
    for i in range(sg_num):
        sg = []
        for idx, duration in enumerate(duration_list):
            if duration[0] <= i < duration[1]:
                sg.append(action_list[idx])
        dsg.append(sg)
    return dsg


def extract_prediction_for_scene_imagination(input_string):
    # Output (imagined):
    # (woman, in, room), (woman, wearing, scarf), (lady, hanging, bag), (bag, cloth),
    # (woman, holding, fitness mat), (fitness mat, in, green)

    # Regular expression pattern to match category and bounding box
    pattern = r'(\b(\w+\s*\w*)\s*, \b(\w+\s*\w*)\s*, \b(\w+\s*\w*)\s*)'

    # Find all matches of the pattern in the input string
    matches = re.findall(pattern, input_string)

    # Extract sg from the matches
    sg = []
    for match in matches:
       sg.append((match[0], match[1], match[2]))

    # Return the enriched scene graph
    return sg


def extract_prediction_for_scene_polishment(input_string):
    # Output (polished):
    # 1st SG: (woman, in, room), (woman, wearing, scarf), (woman, holding, bag), (bottle, in,
    # bag), (water, in, bottle), (bag, cloth).
    # 2nd SG: (woman, in, room), (woman, wearing, scarf), (woman, hanging, bag), (bag, cloth),
    # (woman, holding, fitness mat), (fitness mat, in, green).
    # 3rd SG: (woman, wearing, scarf), (woman, hanging, bag), (bag, cloth), (woman, holding,
    # fitness mat), (fitness mat, in, green), (woman, get out of, office).
    dsg = []
    sents = input_string.split('\n')
    for s in sents:
        sg = extract_prediction_for_scene_imagination(s)
        dsg.append(sg)
    # Return the polished dynamic scene graph
    return dsg