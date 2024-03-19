
def add_prefix_for_action_planning(example, query):
    prompt = ('Now you are an action planner, you will extract event triplets from a text, each triplet in a format “(agent, event-predicate, target, (start-time, end-time))”. '
              '“agent” is the action performer, “target” is the action recipient, and “event-predicate” is the main action.'
              '“start-time” and “end-time” indicate the possible event occurrence order and duration with the basic time interval v.'
              'Next, I will give you several examples for you to understand this task.'
              f'\n{example}\n{query}')
    return prompt


def add_prefix_for_scene_imagination(example, query):
    prompt = ('Now use your imagination, and add the scene structures of the given scene graph (SG) with more possible details.'
              'Enrich it while carefully considering the raw sentence, and the contexts of previous, current and Future SGs.'
              'Next, I will give you several examples for you to understand this task.'
              f'\n{example}\n{query}'
              )
    return prompt


def add_prefix_for_scene_polishment(example, query):
    prompt = (
        'Now use your imagination, and add the scene structures of the given scene graph (SG) with more possible details.'
        'Enrich it while carefully considering the raw sentence, and the contexts of previous, current and Future SGs.'
        'Next, I will give you several examples for you to understand this task.'
        f'\n{example}\n{query}'
        )
    return prompt


def build_prompt_for_action_planning(shot_cand, test_example, args):
    cap = test_example['captions']
    in_context_str = ''
    for i, cur_cand in enumerate(shot_cand):
        cap_train = cur_cand['captions']
        # print(j, cap_train, data_raw_train[idx]['name'])
        # print(j, data_raw_train[idx]['name'])
        input_str = '\ninput: ' + cap_train + '\n'

        triplets = cur_cand['triplets']
        durations = cur_cand['durations']
        tri_str_all = ['output: ']
        for jj in range(len(triplets)):
            tri = triplets[jj]  # tuple
            time_span = durations[jj]
            event_triplet_str = tri.append(time_span)
            tri_str_all.append(event_triplet_str)
        tri_str_all = '\n'.join(tri_str_all)
        io_str = input_str + tri_str_all + '\n'
        # print(io_str)
        in_context_str += io_str

    # print(in_context_str)
    query_str = f'input: {cap} (No explanation. Must give an output or try to imagine a possible output even if the given description is incomplete. )'
    # print(query_str)
    prompt_input = add_prefix_for_action_planning(in_context_str, query_str)

    return prompt_input


def build_prompt_for_scene_imagination(shot_cand, test_example, args):
    in_context_str = ''
    for i, cur_cand in enumerate(shot_cand):
        cap_train = cur_cand['captions']
        # print(j, cap_train, data_raw_train[idx]['name'])
        # print(j, data_raw_train[idx]['name'])
        input_str = '\ninput: \n' + 'Sentence: ' + cap_train + '\n'

        enriched_sg = []
        for jj in range(len(cur_cand['dsg'])):
            enriched_sg_str = '\nEnriched last SG: ' + ','.join(enriched_sg)
            current_sg = cur_cand['dsg'][jj]
            current_sg_str = '\nCurrent SG to enrich: ' + ','.join(current_sg)
            if jj+1 < len(cur_cand['dsg']):
                following_sg = cur_cand['dsg'][jj+1]
            else:
                following_sg = []
            following_sg_str = '\nFollowing SG: ' + ','.join(following_sg)
            imagined_str_all = '\noutput: \n' + ','.join(cur_cand['enriched_dsg'][jj])
            enriched_sg = cur_cand['enriched_dsg'][jj]
            io_str = input_str + enriched_sg_str + current_sg_str + following_sg_str + imagined_str_all + '\n'
            in_context_str += io_str

    # print(in_context_str)
    cap = test_example['captions']
    enriched_sg = ','.join(test_example['enriched_sg'])
    current_sg = ','.join(test_example['current_sg'])
    following_sg = ','.join(test_example['following_sg'])
    query_str = '\ninput: \n' + f'Setence: {cap}\n' + f'Enriched last SG: {enriched_sg}' + f'Current SG to enrich: {current_sg}' + f'Following SG: {following_sg}' + '\n'
    # print(query_str)
    prompt_input = add_prefix_for_scene_imagination(in_context_str, query_str)

    return prompt_input


def build_prompt_for_scene_polishment(shot_cand, test_example, args):

    in_context_str = ''
    for i, cur_cand in enumerate(shot_cand):
        cap_train = cur_cand['captions']
        # print(j, cap_train, data_raw_train[idx]['name'])
        # print(j, data_raw_train[idx]['name'])
        input_str = '\ninput: \n' + 'Sentence: ' + cap_train + '\n'

        dsg_str = ''
        for jj in range(len(cur_cand['enriched_dsg'])):
            current_sg = f'\n{jj} SG: ' + ','.join(cur_cand['enriched_dsg'][jj]) + '\n'
            dsg_str += current_sg

        polished_dsg_str = '\nOutput: \n'
        for jj in range(len(cur_cand['polished_dsg'])):
            current_sg = f'\n{jj} SG: ' + ','.join(cur_cand['polished_dsg'][jj]) + '\n'
            polished_dsg_str += current_sg
        io_str = input_str + dsg_str + polished_dsg_str + '\n'
        in_context_str += io_str

    # print(in_context_str)
    cap = test_example['captions']
    dsg_str = ''
    for jj in range(len(test_example['enriched_dsg'])):
        current_sg = f'\n{jj} SG: ' + ','.join(test_example['enriched_dsg'][jj]) + '\n'
        dsg_str += current_sg

    query_str = f'input: \n' + f'Sentence: {cap}\n' + dsg_str
    # print(query_str)
    prompt_input = add_prefix_for_scene_polishment(in_context_str, query_str)

    return prompt_input
