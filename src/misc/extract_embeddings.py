import argparse
import pickle
import numpy as np
import os

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
    args = parser.parse_args()

    memory_filename = args.memory
    output_filename = args.output
    extract_embeddings(memory_filename, output_filename)
