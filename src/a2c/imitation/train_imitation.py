import os
import pickle
import argparse

import numpy as np
import tensorflow as tf

from src.a2c.models.lstm_actor_critic_model import LSTMActorCriticModel

def imitate(memory_path=None,
            prior_memory_path=None,
            output_dir=None,
            model=None,
            prior=None,
            opt=None,
            train_steps=1000,
            batch_size=40,
            kl_reg=0.01,
            num_actions=6,
            checkpoint_period=50):
    # Load human input from pickle file and concatenate together
    data_file = open(memory_path, 'rb')
    memory_list = pickle.load(data_file)
    data_file.close()

    data_file = open(prior_memory_path, 'rb')
    prior_memory_list = pickle.load(data_file)
    data_file.close()
    print("Loaded files!")

    mem_obs, mem_actions, mem_rewards, mem_dones = [], [], [], []
    for memory in memory_list:
        mem_obs += memory.obs
        mem_actions += memory.actions
        mem_rewards += memory.rewards
        mem_dones += memory.dones
    counts = np.zeros(num_actions)
    for a in mem_actions:
        if a < num_actions: counts[a] += 1
    counts = [(sum(counts) - c) / sum(counts) for c in counts]

    prior_mem_obs, prior_mem_actions, prior_mem_rewards, prior_mem_dones = [], [], [], []
    for memory in prior_memory_list:
        prior_mem_obs += memory.obs
        prior_mem_actions += memory.actions
        prior_mem_rewards += memory.rewards
        prior_mem_dones += memory.dones

    # Build generator that spits out sequences
    def build_gen(g_mem_obs, g_mem_actions, g_mem_rewards, g_mem_dones):
        start = 0
        l = len(g_mem_obs)
        while True:
            if start + batch_size >= l:
                end = batch_size - (l-start)
                yield (g_mem_obs[start:l] + g_mem_obs[0:end],
                       g_mem_actions[start:l] + g_mem_actions[0:end],
                       g_mem_rewards[start:l] + g_mem_rewards[0:end],
                       g_mem_dones[start:l] + g_mem_dones[0:end])
            else:
                end = start + batch_size
                yield (g_mem_obs[start:end],
                       g_mem_actions[start:end],
                       g_mem_rewards[start:end],
                       g_mem_dones[start:end])
            start = end
    imitation_generator = build_gen(mem_obs, mem_actions, mem_rewards, mem_dones)
    prior_generator = build_gen(prior_mem_obs, prior_mem_actions, prior_mem_rewards, prior_mem_dones)

    small_value = 0.0000001
    states = np.zeros((2, model.lstm_size * 2)).astype(np.float32)
    prior_states = np.zeros((2, model.lstm_size * 2)).astype(np.float32)
    print("Starting steps...")
    for train_step in range(train_steps):
        # Combine prior and imitation batches
        obs, actions, rewards, dones  = next(imitation_generator)
        weights = [counts[action] for action in actions]
        p_obs, p_actions, p_rewards, p_dones  = next(prior_generator)
        obs = np.asarray(obs + p_obs)
        actions = np.asarray(actions + p_actions)
        rewards = np.asarray(rewards + p_rewards)
        dones = np.asarray(dones + p_dones)

        obs = model.process_inputs(obs)
        with tf.GradientTape() as tape:
            logits, values, states = model([obs, states, dones])
            logits = tf.nn.softmax(logits)

            # Calculate cross entropy loss            
            scce = tf.keras.losses.SparseCategoricalCrossentropy()
            scce_loss = scce(actions[0:batch_size], logits[0:batch_size], sample_weight=weights)
            
            # Calculate KL loss
            prior_logits, _, prior_states = prior([obs, prior_states, dones])
            prior_logits = tf.nn.softmax(prior_logits)
            kl_loss = tf.reduce_mean(prior_logits *
                                     tf.math.log(prior_logits / (logits + small_value) + small_value))
            
            total_loss = scce_loss + (kl_reg * kl_loss)

        # Calculate and apply gradients
        total_grads = tape.gradient(total_loss, model.trainable_weights)
        opt.apply_gradients(zip(total_grads, model.trainable_weights))

        print("Step: {}".format(train_step, total_loss))
        print("Total Loss: {} | SCCE Loss: {} | KL Loss: {}".format(total_loss, scce_loss, kl_loss))
        if train_step % 10 == 0:
            predict_actions = [np.argmax(distribution) for distribution in logits]
            correct = [1 if t == p else 0 for t, p in zip(actions, predict_actions)]
            print("Accuracy: {}".format(sum(correct) / len(correct)))
        if train_step % checkpoint_period == 0:
            _save_path = os.path.join(output_dir, "imitation_model_{}.h5".format(train_step))
            model.save_weights(_save_path)
            print("Checkpoint saved to {}".format(_save_path))
        print("=" * 20)

    _save_path = os.path.join(output_dir, "imitation_model.h5")
    model.save_weights(_save_path)
    print("Checkpoint saved to {}".format(_save_path))

def main(args,
         train_steps=2500,
         learning_rate=0.00042,
         kl_reg=0.01,
         num_steps=40,
         num_actions=6,
         stack_size=1,
         actor_fc=[128],
         critic_fc=[128],
         before_fc=[256],
         lstm_size=256,
         conv_size="quake",
         checkpoint_period=100):
    if args.restore == None:
        raise ValueError('Restore must be specified')
    if args.memory_path == None:
        raise ValueError('Memory path must be specified')

    prior = LSTMActorCriticModel(num_actions=num_actions,
                                      state_size=[84,84,3],
                                      stack_size=1,
                                      num_steps=num_steps,
                                      num_envs=2,
                                      actor_fc=actor_fc,
                                      critic_fc=critic_fc,
                                      before_fc=before_fc,
                                      lstm_size=lstm_size,
                                      conv_size=conv_size,
                                      retro=True)
    prior.load_weights(args.restore)

    # Build model
    model = LSTMActorCriticModel(num_actions=num_actions,
                                      state_size=[84,84,3],
                                      stack_size=1,
                                      num_steps=num_steps,
                                      num_envs=2,
                                      actor_fc=actor_fc,
                                      critic_fc=critic_fc,
                                      before_fc=before_fc,
                                      lstm_size=lstm_size,
                                      conv_size=conv_size,
                                      retro=True)
    model.load_weights(args.restore)

    opt = tf.optimizers.Adam(learning_rate)
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

    imitate(memory_path=args.memory_path,
            prior_memory_path=args.prior_memory_path,
            output_dir=args.output_dir,
            model=model,
            prior=prior,
            opt=opt,
            train_steps=train_steps,
            batch_size=num_steps,
            kl_reg=kl_reg,
            checkpoint_period=checkpoint_period)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Prierarchy Imitation Learning')
    parser.add_argument('--memory-path', type=str, default=None)
    parser.add_argument('--prior-memory-path', type=str, default=None)
    parser.add_argument('--restore', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='/tmp/prierarchy_imitation')
    args = parser.parse_args()

    main(args)
