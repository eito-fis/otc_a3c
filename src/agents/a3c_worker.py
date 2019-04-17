import os
import pickle
import argparse

import threading
import multiprocessing

import numpy as np
import tensorflow as tf

from queue import Queue
from collections import Counter

from src.models.actor_critic_model import ActorCriticModel, Memory

def record(episode,
           episode_reward,
           episode_floor,
           worker_idx,
           global_ep_reward,
           global_ep_floor,
           result_queue,
           total_loss,
           num_steps,
           explained_variance,
           entropy_loss):
    """
    Stores score and print statistics.
    Arguments:
    episode: Current episode
    episode_reward: Total reward for current episode
    worker_idx: ID of worker being recorded
    global_ep_reward: The moving average of the global reward
    result_queue: Queue storing the moving average of the scores
    total_loss: Total loss accumualted for current episode
    num_steps: Total steps for current episode
    """
    # Update global moving averages
    if global_ep_reward == 0:
        global_ep_reward = episode_reward
    else:
        global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
    if global_ep_floor == 0:
        global_ep_floor = episode_floor
    else:
        global_ep_floor = global_ep_floor * 0.8 + episode_floor * 0.2

    # Print metrics
    print("Episode: {} | Moving Average Reward: {} | Episode Reward: {} | Moving Average Floor: {} | Episode Floor: {} | Loss: {} | Explained Variance: {} | Steps: {} | Worker: {} | Entropy Loss: {}".format(episode, global_ep_reward, episode_reward, global_ep_floor, episode_floor, int(total_loss * 1000) / 1000, explained_variance, num_steps, worker_idx, entropy_loss))

    # Add metrics to queue
    result_queue.put(global_ep_reward)

    return global_ep_reward, global_ep_floor

class Worker(threading.Thread):
    '''
    Worker agent
    - Asynchronously plays episodes and calculates gradients, before applying the gradient to the 
      global model
    Arguments:
    idx: Unique worker id
    result_queue: Queue object that allows the child threads to talk to the parent
    global_model: Global model to pull weights from and apply gradients to
    local_model: Local model that will play episdoes and calculate gradients
    max_episodes: Total amount of episodes to be played be agents
    num_actions: Number of possible actions in the environment
    state_size: Expected size of state returned from environment
    stack_size: Number of stacked frames our model will consider
    env_func: Callable function that builds an environment for each worker. Will be passed an idx
    opt: Tensorflow optimizer object
    gamma: Decay coefficient used to discount future reward while calculating loss
    entropy_discount: Discount coefficient used to control our entropy loss
    value_discount: Discount coefficient used to control our critic model loss
    boredom_thresh: Threshold for the standard deviation of previous frames - controls how easily
        our model gets bored
    update_freq: Number of time steps each worker takes before calcualting gradient and updating
        global model
    summary_writer: Summary_Writer object used to write tensorboard logs
    load_path: Path to model weights to be loaded
    memory_path: Path to where memories should be saved. If None, no memories are saved
    save_path: Path to where model weights should be saved
    log_period: Number of epochs to wait before writing log information
    checkpoint_period: Numer of epochs to wait before saving model
    memory_period: Number of epochs to wait before saving a memory
    '''
    # Keeps track of all episodes played by all workers
    global_episode = 0
    # Moving average reward
    global_moving_average_reward = 0.0
    global_moving_average_floor = 0.0
    best_score = 0
    save_lock = threading.Lock()

    def __init__(self,
                 idx=0,
                 result_queue=None,
                 global_model=None,
                 local_model=None,
                 max_episodes=500,
                 num_actions=2,
                 state_size=[4],
                 stack_size=None,
                 env_func=None,
                 opt=None,
                 gamma=0.99,
                 entropy_discount=None,
                 value_discount=None,
                 boredom_thresh=None,
                 update_freq=None,
                 summary_writer=None,
                 memory_path=None,
                 save_path='/tmp/a3c/workers',
                 log_period=10,
                 checkpoint_period=10,
                 memory_period=None):
        super(Worker, self).__init__()
        self.worker_idx = idx
        self.result_queue = result_queue
        self.global_model = global_model

        # Create local model, then copy the global models weights
        self.local_model = local_model
        self.local_model.set_weights(self.global_model.get_weights())

        self.max_episodes = max_episodes
        self.num_actions = num_actions
        self.state_size = state_size
        self.stack_size = stack_size
        
        # Build the environment!
        self.env = env_func(idx)
        self.opt = opt

        self.gamma = gamma
        self.entropy_discount = entropy_discount
        self.value_discount = value_discount
        self.boredom_thresh = boredom_thresh
        self.update_freq = update_freq

        self.summary_writer = summary_writer
        self.memory_path = memory_path
        self.save_path = save_path
        self.log_period = log_period
        self.checkpoint_period = checkpoint_period
        self.memory_period = memory_period

        self.ep_loss = 0.0

    def run(self):
        '''
        The main loop ran by each worker
        - Takes steps until either update freq steps have been taken, or the episode ends. Then,it calculates
          a gradient relative to the local model, and applies it to the global model. Finally, it pulls the global
          model weights, and continues taking steps.
        '''
        total_step = 1
        mem = Memory()

        # Cotinue running while total worker episdoes is less than max episodes
        while Worker.global_episode < self.max_episodes:
            # Set initial states
            state, obs = self.env.reset()
            mem.clear()
            rolling_average_state = state
            current_episode = Worker.global_episode
            save_visual = (self.memory_path != None and current_episode % self.visual_period == 0)
            ep_reward = 0.
            ep_rewards = []
            ep_values = []
            ep_steps = 0
            self.ep_loss = 0

            action = 0
            floor = 0.
            time_count = 0
            done = False
            # Initialize our stack memory with random values
            prev_states = [np.random.random(state.shape).astype(np.float32) for _ in range(self.stack_size)]

            # Play an episode!
            while not done:
                # Update our stack memory with our current state
                prev_states = prev_states[1:] + [state]

                # Calculate the deviation between the rolling average of previous frames and the current frame
                # Allows us to detect is we aren't reasonably moving, and take action
                _deviation = tf.reduce_sum(tf.math.squared_difference(rolling_average_state, state))
                if time_count > 10 and _deviation < self.boredom_thresh:
                    # Take an action that isn't the previous action
                    possible_actions = np.delete(np.array(range(self.num_actions)), action)
                    action = np.random.choice(possible_actions)
                else:
                    # Convert our stack memory into a single vector
                    stacked_state = np.concatenate(prev_states, axis=-1)
                    # Pass vector to model and sample from the logits to get a single action
                    action, _ = self.local_model.get_action_value(stacked_state[None, :],
                                                                  np.array([floor], dtype=np.float32)[None, :])

                # Take our step
                (new_state, reward, done, _), new_obs  = self.env.step(action)

                # Save data to memory. Only save extra metrics if saving memory in the end
                mem.store(state, action, reward, floor)
                if save_visual: mem.obs.append(obs)
                ep_reward += reward

                # If reward is 1, we found an exit and moved up a floor
                if reward >= 1: floor += 1

                # Calculate a gradient if we've hit update_freq many steps or have reached the end of an episode
                if time_count == self.update_freq or done:
                    # Calculate loss based on trajectory
                    with tf.GradientTape() as tape:
                        total_loss, values, rewards, entropy_loss  = self.compute_loss(mem, done, self.gamma, save_visual, stacked_state, floor)
                    ep_values.extend(values.tolist())
                    ep_rewards.extend(rewards)
                    self.ep_loss += total_loss

                    # Calculate gradient for local model
                    total_grads = tape.gradient(total_loss, self.local_model.trainable_weights)
                    # Apply gradient to global model
                    self.opt.apply_gradients(zip(total_grads, self.global_model.trainable_weights))
                    # Update local model with new global model weights
                    self.local_model.set_weights(self.global_model.get_weights())
                    
                    if done:
                        self.log_episode(save_visual, current_episode, ep_steps, ep_reward, mem, total_loss, ep_values, ep_rewards, entropy_loss)
                        Worker.global_episode += 1
                    time_count = 0

                    mem.clear()
                else:
                    # Update values for next iteration
                    ep_steps += 1
                    time_count += 1
                    total_step += 1
                    obs = new_obs
                    state = new_state
                    rolling_average_state = rolling_average_state * 0.8 + new_state * 0.2
        # Once max_episode amount of episode have been played, return None as a single to end the program
        self.result_queue.put(None)

    def compute_loss(self,
                     memory,
                     done,
                     gamma,
                     save_visual,
                     stacked_state,
                     final_floor):
        '''
        Computes loss for A2C model
        Arguments:
        memory: The memory of the trajectory the loss is being computed for
        done: Whether or not we ended because of update freq, or because an episode ended
        gamma: Coefficient to discount future reward by
        save_visual: Whether or not we are saving memories
        stacked_state: The final stacked state
        '''
        # If not done, estimate the future discount reward by calling the critic model on
        # the final state
        if done:
            reward_sum = 0.
        else:
            reward_sum = self.local_model.critic_model([stacked_state[None, :],
                                                        np.array([final_floor], dtype=np.float32)[None, :]])
            reward_sum = np.squeeze(reward_sum.numpy())

        # Calculate discounted rewards
        discounted_rewards = []
        for reward in memory.rewards[::-1]:
            if reward == 0:
                reward_sum = reward + gamma * reward_sum
            else:
                reward_sum = reward
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        # Build batch of stacked states
        stacked_states = []
        prev_states = [np.random.random(tuple(self.state_size)) for _ in range(self.stack_size)]
        for index, (state, action) in enumerate(zip(memory.states, memory.actions)):
            prev_states = prev_states[1:] + [state]
            stacked_states.append(np.concatenate(prev_states, axis=-1))

        # Get batch of logits and values
        logits, values = self.local_model([np.array(stacked_states, dtype=np.float32),
                                           np.array(memory.floors, dtype=np.float32)[:, None]])

        # Calculate our advantages
        advantage = np.array(discounted_rewards)[:, None] - values

        # Calculate our entropy loss - returns a single value
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        entropy_loss = cce(tf.stop_gradient(tf.nn.softmax(logits)), logits) * -1

        # Calculate our policy loss - returns a single value
        wce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        policy_loss = wce(memory.actions,
                          logits,
                          sample_weight=tf.stop_gradient(advantage))

        # Calculate our value loss - returns a single value
        mse = tf.keras.losses.MeanSquaredError()
        value_loss = mse(tf.stop_gradient(np.array(discounted_rewards)[:, None]), values)

        if save_visual:
            memory.probs.extend(np.squeeze(tf.nn.softmax(logits).numpy()).tolist())
            memory.values.extend(np.squeeze(values.numpy()).tolist())

        # Discount losses, add together and return
        return policy_loss + \
               (value_loss * self.value_discount) + \
               (entropy_loss * self.entropy_discount), np.squeeze(values.numpy()), discounted_rewards, entropy_loss

    def log_episode(self, save_visual, current_episode, ep_steps, ep_reward, mem, total_loss, ep_values, ep_rewards, entropy_loss):
        '''
        Helper function that logs and saves info
        '''
        # Save the memory of our episode
        if save_visual:
            pickle_path = os.path.join(self.memory_path, "memory_{}_{}".format(current_episode, self.worker_idx))
            os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
            pickle_file = open(pickle_path, 'wb+')
            pickle.dump(mem, pickle_file)
            pickle_file.close()
            print("Memory saved to {}".format(pickle_path))

        # Metrics logging and saving
        ep_floor = mem.floors[-1]
        explained_variance = self.explained_variance(ep_values, ep_rewards)
        Worker.global_moving_average_reward, Worker.global_moving_average_floor = record(Worker.global_episode,
                                                                                         ep_reward,
                                                                                         ep_floor,
                                                                                         self.worker_idx,
                                                                                         Worker.global_moving_average_reward,
                                                                                         Worker.global_moving_average_floor,
                                                                                         self.result_queue,
                                                                                         self.ep_loss,
                                                                                         ep_steps,
                                                                                         explained_variance,
                                                                                         entropy_loss)
        print(ep_values)

        # We must use a lock to save our model and to print to prevent data races.
        if ep_reward > Worker.best_score:
            with Worker.save_lock:
                print("Saving best model to {}, "
                    "episode score: {}".format(self.save_path, ep_reward))
                self.global_model.save_weights(
                    os.path.join(self.save_path, 'best_model.h5')
                )
                Worker.best_score = ep_reward

        # Save model and logs for tensorboard
        if current_episode % self.log_period == 0:
            with self.summary_writer.as_default():
                tf.summary.scalar("Episode Reward", ep_reward, current_episode)
                tf.summary.scalar("Moving Global Average Reward", Worker.global_moving_average_reward, current_episode)
                tf.summary.scalar("Episode Floor", ep_floor, current_episode)
                tf.summary.scalar("Moving Global Average Floor", Worker.global_moving_average_floor, current_episode)
                tf.summary.scalar("Explained Variance", explained_variance, current_episode)
                tf.summary.scalar("Episode Loss", tf.reduce_sum(total_loss), current_episode)
        if current_episode % self.checkpoint_period == 0:
            _save_path = os.path.join(self.save_path, "worker_{}_model_{}.h5".format(self.worker_idx, current_episode))
            self.local_model.save_weights(_save_path)
            print("Checkpoint saved to {}".format(_save_path))

    def explained_variance(self, y_pred, y_true):
        """
        Computes fraction of variance that ypred explains about y.
        Returns 1 - Var[y-ypred] / Var[y]
        interpretation:
            ev=0  =>  might as well have predicted zero
            ev=1  =>  perfect prediction
            ev<0  =>  worse than just predicting zero
        """
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        assert y_true.ndim == 1 and y_pred.ndim == 1
        var_y = np.var(y_true)
        return 0 if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

