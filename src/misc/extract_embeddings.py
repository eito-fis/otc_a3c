import argparse
import pickle
import numpy as np
import os

def extract_embeddings_actions(memory_filename, output_filename):
    with open(memory_filename, 'rb') as mf:
        memory = pickle.load(mf)

    embeddings = []
    actions = []

    if type(memory) == type([]) and hasattr(memory[0], 'states'):
        for m in memory:
            if len(m.states) == 0: continue
            embeddings.append(np.array([np.array(s) for s in m.states]))
            actions.append(np.array(m.actions))
    elif hasattr(memory, 'states'):
        embeddings.append(np.array([np.array(s) for s in memory.states]))
        actions.append(np.array(memory.actions))
    else:
        raise Exception('Provide human replay or memory file as input.')

    print('Embeddings shapes: {}'.format(', '.join([str(ems.shape) for ems in embeddings])))
    print('Actions shapes: {}'.format(', '.join([str(acts.shape) for acts in actions])))
    print('Saving dictionary with keys "embeddings", "actions"')

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, 'wb') as of:
        pickle.dump({'embeddings': embeddings, 'actions': actions}, of)
    print('Saved!')


def extract_paired_embeddings(memory_filename, output_filename, frames):
    with open(memory_filename, 'rb') as mf:
        memory = pickle.load(mf)

    paired_embeddings = []

    if type(memory) == type([]) and hasattr(memory[0], 'states'):
        for m in memory:
            embeddings = [np.array(s) for s in m.states]
            if frames > 0:
                paired_embeddings.extend(list(zip(embeddings[:-frames], embeddings[frames:])))
            else:
                paired_embeddings.extend(list(zip(embeddings, embeddings)))
    elif hasattr(memory, 'states'):
        embeddings = [np.array(s) for s in memory.states]
        if frames > 0:
            paired_embeddings = list(zip(embeddings[:-frames], embeddings[frames:]))
        else:
            paired_embeddings = list(zip(embeddings, embeddings))
    else:
        raise Exception('Provide human replay or memory file as input.')

    paired_embeddings = np.array(paired_embeddings)

    print('Paired embeddings shape: {}'.format(paired_embeddings.shape))
    print('Saving numpy array of paired embeddings')

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, 'wb') as of:
        pickle.dump(paired_embeddings, of)
    print('Saved!')

def extract_embeddings(memory_filename, output_filename):
    with open(memory_filename, 'rb') as mf:
        memory = pickle.load(mf)

    embeddings = []
    images = []
    actions = []

    if type(memory) == type([]) and hasattr(memory[0], 'states'):
        for m in memory:
            embeddings.extend([np.array(s) for s in m.states])
            images.extend([np.array(i) for i in m.obs])
            actions.extend(m.actions)
    elif hasattr(memory, 'states'):
        embeddings = [np.array(s) for s in memory.states]
        images = [np.array(i) for i in memory.obs]
        actions = list(memory.actions)
    else:
        raise Exception('Provide human replay or memory file as input.')

    embeddings = np.array(embeddings)
    images = np.array(images)
    actions = np.array(actions)

    print('Embeddings shape: {}'.format(embeddings.shape))
    print('Images shape: {}'.format(images.shape))
    print('Actions shape: {}'.format(actions.shape))
    print('Saving dictionary with keys "embeddings", "images", "actions"')

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, 'wb') as of:
        pickle.dump({'embeddings': embeddings, 'images': images, 'actions': actions}, of)
    print('Saved!')

if __name__ == '__main__':
    #PARSE COMMAND-LINE ARGUMENTS#
    parser = argparse.ArgumentParser('extract embeddings from memory file')
    parser.add_argument('--output', type=str, default='./embeddings/extracted')
    parser.add_argument('--memory', type=str, required=True)
    parser.add_argument('--paired', action='store_true')
    parser.add_argument('--actions', action='store_true')
    parser.add_argument('--frames', type=int, default=1, help='frames into future for paired embeddings')
    args = parser.parse_args()

    memory_filename = args.memory
    output_filename = args.output
    frames = args.frames
    if args.actions:
        extract_embeddings_actions(memory_filename, output_filename)
    elif args.paired:
        extract_paired_embeddings(memory_filename, output_filename, frames)
    else:
        extract_embeddings(memory_filename, output_filename)
