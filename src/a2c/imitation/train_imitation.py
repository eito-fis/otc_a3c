import os
import pickle
import argparse

import numpy as np
import tensorflow as tf

from src.a2c.models.lstm_actor_critic_model import LSTMActorCriticModel
from src.a2c.eval.lstm_eval import Memory

def imitate(memory_path=None,
            prior_memory_path=None,
            output_dir=None,
            model=None,
            prior=None,
            opt=None,
            train_steps=1000,
            batch_size=40,
            kl_reg=0.01,
            num_mem=10,
            num_val=10,
            num_prior=10,
            num_actions=6,
            checkpoint_period=100,
            wandb=None):
    # Load human input from pickle file and concatenate together
    data_file = open(memory_path, 'rb')
    memory_list = pickle.load(data_file)
    data_file.close()

    data_file = open(prior_memory_path, 'rb')
    prior_memory_list = pickle.load(data_file)
    data_file.close()
    print("Loaded files!")

    validation_len = len(memory_list) // 10

    mem_obs, mem_actions, mem_rewards, mem_dones = [], [], [], []
    for memory in memory_list[:-validation_len]:
        if 6 in memory.actions: continue
        mem_obs += memory.obs
        mem_actions += memory.actions
        mem_rewards += memory.rewards
        mem_dones += memory.dones
    counts = np.zeros(num_actions)
    for a in mem_actions:
        if a < num_actions: counts[a] += 1
    counts = [(sum(counts) - c) / sum(counts) for c in counts]

    val_mem_obs, val_mem_actions, val_mem_rewards, val_mem_dones = [], [], [], []
    for memory in memory_list[-validation_len:]:
        if 6 in memory.actions: continue
        val_mem_obs += memory.obs
        val_mem_actions += memory.actions
        val_mem_rewards += memory.rewards
        val_mem_dones += memory.dones

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

    # Build generator that maintains multiple prior sequences
    def build_multi_gen(g_p_mem_obs, g_p_mem_actions, g_p_mem_rewards, g_p_mem_dones, num_gen):
        prior_gens = []
        done_index = [i for i, d in enumerate(g_p_mem_dones) if d == 1.0] + [None]
        num_dones = len(done_index) - 1
        if num_gen > num_dones != 0:
            raise ValueError("Number of dones ({}) must\
                              be greater than num_gen ({})".format(num_dones, num_gen))

        num_long_seq = num_dones % num_gen
        num_short_seq = num_gen - num_long_seq

        len_short_seq = num_dones // num_gen
        len_long_seq = len_short_seq + 1

        num_short_done = num_short_seq * len_short_seq
        num_long_done = num_long_seq * len_long_seq
        for i in range(0, num_short_done, len_short_seq):
            prior_gens.append(build_gen(g_p_mem_obs[done_index[i]:done_index[i + len_short_seq]],
                                        g_p_mem_actions[done_index[i]:done_index[i + len_short_seq]],
                                        g_p_mem_rewards[done_index[i]:done_index[i + len_short_seq]],
                                        g_p_mem_dones[done_index[i]:done_index[i + len_short_seq]]))
        for i in range(num_short_done, num_short_done + num_long_done, len_long_seq):
            prior_gens.append(build_gen(g_p_mem_obs[done_index[i]:done_index[i + len_long_seq]],
                                        g_p_mem_actions[done_index[i]:done_index[i + len_long_seq]],
                                        g_p_mem_rewards[done_index[i]:done_index[i + len_long_seq]],
                                        g_p_mem_dones[done_index[i]:done_index[i + len_long_seq]]))
        while True:
            y_mem_obs, y_mem_actions, y_mem_rewards, y_mem_dones = [], [], [], []
            for g in prior_gens:
                g_y_mem_obs, g_y_mem_actions, g_y_mem_rewards, g_y_mem_dones = next(g)
                y_mem_obs += g_y_mem_obs
                y_mem_actions += g_y_mem_actions
                y_mem_rewards += g_y_mem_rewards
                y_mem_dones += g_y_mem_dones
            yield (y_mem_obs, y_mem_actions, y_mem_rewards, y_mem_dones)
            

    imitation_generator = build_multi_gen(mem_obs, mem_actions, mem_rewards, mem_dones, num_mem)
    validation_generator = build_multi_gen(val_mem_obs, val_mem_actions, val_mem_rewards, val_mem_dones, num_val)
    prior_generator = build_multi_gen(prior_mem_obs, prior_mem_actions, prior_mem_rewards, prior_mem_dones, num_prior)

    small_value = 0.0000001
    states = np.zeros((num_mem + num_val + num_prior, model.lstm_size * 2)).astype(np.float32)
    prior_states = np.zeros((num_mem + num_val + num_prior, model.lstm_size * 2)).astype(np.float32)
    print("Starting steps...")
    for train_step in range(train_steps):
        # Combine prior and imitation batches
        obs, actions, rewards, dones  = next(imitation_generator)
        weights = [counts[action] for action in actions]
        v_obs, v_actions, v_rewards, v_dones  = next(validation_generator)
        p_obs, p_actions, p_rewards, p_dones  = next(prior_generator)

        obs = obs + p_obs + v_obs
        obs = model.process_inputs(obs)
        actions = np.asarray(actions + p_actions + v_actions).astype(np.float32)
        rewards = np.asarray(rewards + p_rewards + v_rewards).astype(np.float32)
        dones = np.asarray(dones + p_dones + v_dones).astype(np.float32)

        with tf.GradientTape() as tape:
            logits, values, states = model([obs, states, dones])
            logits = tf.nn.softmax(logits)

            # Calculate cross entropy loss
            scce = tf.keras.losses.SparseCategoricalCrossentropy()
            scce_loss = scce(actions[0:batch_size * num_mem], logits[0:batch_size * num_mem], sample_weight=weights)
            
            # Calculate KL loss
            prior_logits, _, prior_states = prior([obs, prior_states, dones])
            prior_logits = tf.nn.softmax(prior_logits)
            kl_loss = tf.reduce_mean(prior_logits[:-batch_size * num_val] *
                                     tf.math.log(prior_logits[:-batch_size * num_val] /
                                                 (logits[:-batch_size * num_val] + small_value)
                                                 + small_value))
            
            total_loss = scce_loss + (kl_reg * kl_loss)

        # Calculate and apply gradients
        total_grads = tape.gradient(total_loss, model.trainable_weights)
        opt.apply_gradients(zip(total_grads, model.trainable_weights))

        predict_actions = [np.argmax(distribution) for distribution in logits[0:batch_size * num_mem]]
        correct = [1 if t == p else 0 for t, p in zip(actions[0:batch_size * num_mem], predict_actions)]
        accuracy = sum(correct) / len(correct)

        v_predict_actions = [np.argmax(distribution) for distribution in logits[-batch_size * num_val:]]
        v_correct = [1 if t == p else 0 for t, p in zip(actions[-batch_size * num_val:], v_predict_actions)]
        v_accuracy = sum(v_correct) / len(v_correct)

        print("Step: {}".format(train_step, total_loss))
        print("Total Loss: {} | SCCE Loss: {} | KL Loss: {}".format(total_loss, scce_loss, kl_loss))
        print("Accuracy: {} | Validation Accuracy {}".format(accuracy, v_accuracy))

        logging_period = 1
        if train_step % logging_period == 0:
            if wandb != None:
                wandb.log({"Train Step": train_step,
                                "Accuracy": accuracy,
                                "Validation Accuracy": v_accuracy,
                                "Total Loss": total_loss.numpy(),
                                "SCCE Loss": scce_loss.numpy(),
                                "KL Loss": kl_loss.numpy()})
        if train_step % checkpoint_period == 0:
            _save_path = os.path.join(output_dir, "imitation_model_{}.h5".format(train_step))
            model.save_weights(_save_path)
            print("Checkpoint saved to {}".format(_save_path))
        print("=" * 20)

    _save_path = os.path.join(output_dir, "imitation_model.h5")
    model.save_weights(_save_path)
    print("Checkpoint saved to {}".format(_save_path))

def main(args,
         train_steps=10000,
         learning_rate=0.00042,
         kl_reg=10,
         num_mem=5,
         num_val=1,
         num_prior=5,
         num_steps=50,
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
                                      num_envs=num_prior,
                                      actor_fc=actor_fc,
                                      critic_fc=critic_fc,
                                      before_fc=before_fc,
                                      lstm_size=lstm_size,
                                      conv_size=conv_size,
                                      retro=True)
    if args.prior is not None:
        prior.load_weights(args.prior)
    else:
        prior.load_weights(args.restore)

    # Build model
    model = LSTMActorCriticModel(num_actions=num_actions,
                                      state_size=[84,84,3],
                                      stack_size=1,
                                      num_steps=num_steps,
                                      num_envs=num_prior,
                                      actor_fc=actor_fc,
                                      critic_fc=critic_fc,
                                      before_fc=before_fc,
                                      lstm_size=lstm_size,
                                      conv_size=conv_size,
                                      retro=True)
    model.load_weights(args.restore)

    opt = tf.optimizers.Adam(learning_rate)
    os.makedirs(os.path.dirname(args.output_dir + "/test"), exist_ok=True)

    if args.wandb:
        import wandb
        if args.wandb_name != None:
            wandb.init(name=args.wandb_name,
                       project="obstacle-tower-challenge-supervised",
                       entity="42 Robolab")
        else:
            wandb.init(project="obstacle-tower-challenge-supervised",
                       entity="42 Robolab")
            wandb.config.update({"Human Memory Path": args.memory_path,
                                 "Prior Memory Path": args.prior_memory_path,
                                 "Restore": args.restore,
                                 "Train Steps": train_steps,
                                 "KL Reg": kl_reg,
                                 "Num Prior": num_prior,
                                 "Checkpoint Period": checkpoint_period})
    else: wandb = None

    imitate(memory_path=args.memory_path,
            prior_memory_path=args.prior_memory_path,
            output_dir=args.output_dir,
            model=model,
            prior=prior,
            opt=opt,
            train_steps=train_steps,
            batch_size=num_steps,
            kl_reg=kl_reg,
            num_mem=num_mem,
            num_val=num_val,
            num_prior=num_prior,
            checkpoint_period=checkpoint_period,
            wandb=wandb)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Prierarchy Imitation Learning')
    parser.add_argument('--memory-path', type=str, default=None)
    parser.add_argument('--prior-memory-path', type=str, default=None)
    parser.add_argument('--restore', type=str, default=None)
    parser.add_argument('--prior', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='/tmp/prierarchy_imitation')

    # Wandb flags
    parser.add_argument(
        '--wandb',
        default=False,
        action='store_true')
    parser.add_argument(
        '--wandb-name',
        type=str,
        default=None)

    args = parser.parse_args()

    main(args)
