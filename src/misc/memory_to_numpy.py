import numpy as np
import argparse
import pickle
import sys
import os

def convert(memory_filename, output_filename):
    with open(memory_filename, 'rb') as mf:
        memory = pickle.load(mf)

    if type(memory) == type([]):
        memories = memory
    else:
        memories = [memory]

    big_memory_dict = {
        'states': [],
        'actions': [],
        'rewards': [],
        'obs': [],
        'probs': [],
        'values': [],
        'novelty': [],
        'terminals': [], # 1 if last frame in the floor or end of the episode, 0 otherwise
    }
    for i, m in enumerate(memories):
        episode_len = len(m.states)
        sys.stdout.write('Processing memory #{}/{}...\r'.format(i+1, len(memories)))
        sys.stdout.flush()
        if episode_len == 0: continue
        for key in big_memory_dict.keys():
            if key == 'terminals': continue
            values = list(getattr(m, key))[1:] # because of error in first state extraction - (168,3) instead of (168,168,3)
            if key in ['obs', 'states']:
                values = [np.array(v) for v in values]
            big_memory_dict[key].extend(values)
        terminals = [int(r >= 1.) for r in m.rewards[1:-1]] + [1] # also start from second frame
        big_memory_dict['terminals'].extend(terminals)
    print('\nFinished processing memories')

    for key in big_memory_dict.keys():
        big_memory_dict[key] = np.array(big_memory_dict[key])
        print('{}: shape {}'.format(key,big_memory_dict[key].shape))

    print('Saving pickled dictionary with keys {} to {} ...'.format(big_memory_dict.keys(), output_filename))
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, 'wb') as of:
        pickle.dump(big_memory_dict, of)
    print('Done')


if __name__ == '__main__':
    #PARSE COMMAND-LINE ARGUMENTS#
    parser = argparse.ArgumentParser('render video visualizing memory from games')
    parser.add_argument('--memory', type=str, required=True, help='memory filename')
    parser.add_argument('--output', type=str, default='./memories', help='filename of an extracted dict of numpy arrays')
    args = parser.parse_args()

    memory_filename = args.memory
    output_filename = args.output

    convert(memory_filename, output_filename)
