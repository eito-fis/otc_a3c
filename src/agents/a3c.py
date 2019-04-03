
import os
import gc
import pickle
import argparse

import threading
import multiprocessing

import gym
import numpy as np

from queue import Queue

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

from collections import Counter

def record(episode,
           episode_reward,
           worker_idx,
           global_ep_reward,
           result_queue,
           total_loss,
           num_steps):
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
    # Update global moving average
    if global_ep_reward == 0:
        global_ep_reward = episode_reward
    else:
        global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01

    # Print metrics
    print("Episode: {} |\
    Moving Average Reward: {} |\
    Episode Reward: {} |\
    Loss: {} |\
    Steps: {} |\
    Worker: {}".format(episode, global_ep_reward, episode_reward, total_loss, num_steps, worker_idx))

    # Add metrics to queue
    result_queue.put(global_ep_reward)

    return global_ep_reward

class Memory:
    '''
    Stores trajectory information
    - One memory per episode played by the agent
    '''
    def __init__(self):
        # Build storage arrays
        self.states = []
        self.actions = []
        self.rewards = []
        self.obs = []

    def store(self, state, action, reward):
        # Save state - action - reward pairs
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        # Wipe data
        self.states = []
        self.actions = []
        self.rewards = []
        self.obs=[]

class ActorCriticModel(keras.Model):
    '''
    Actor Critic Model
    - Contains both our actor and critic model seperately
    as Keras Sequential models
    Arguments:
    num_actions: Number of logits the actor model will output
    state_size: List containing the expected size of the state
    stack_size: Numer of states we can expect in our stack
    actor_fc: Iterable containing the amount of neurons per layer for the actor model
    critic_fc: Iterable containing the amount of neurons per layer for the critic model
        ex: (1024, 512, 256) would make 3 fully connected layers, with 1024, 512 and 256
            layers respectively
    '''
    def __init__(self,
                 num_actions=None,
                 state_size=None,
                 stack_size=None,
                 actor_fc=None,
                 critic_fc=None):
        super().__init__()

        # Multiply the final dimension of the state by stack_size to get correct input_size
        state_size = state_size[:-1] + [state_size[-1] * stack_size]

        # Build the fully connected layers for the actor and critic models
        self.actor_fc = [keras.layers.Dense(neurons, activation="relu") for neurons in actor_fc]
        self.critic_fc = [keras.layers.Dense(neurons, activation="relu") for neurons in critic_fc]

        # Build the output layers for the actor and critic models
        self.actor_logits = keras.layers.Dense(num_actions, name='policy_logits')
        self.value = keras.layers.Dense(1, name='value', activation='relu')

        # Build the final actor and critic models
        self.actor_model = tf.keras.Sequential(self.actor_fc + [self.actor_logits])
        self.actor_model.build([None] + state_size)
        self.critic_model = tf.keras.Sequential(self.critic_fc + [self.value])
        self.critic_model.build([None] + state_size)

        # Run each model with random inputs to force Keras to build the static graphs
        self.actor_model(np.random.random((1,) + tuple(state_size)))
        self.critic_model(np.random.random((1,) + tuple(state_size)))
        self.predict(np.random.random((1,) + tuple(state_size)))

    def call(self, inputs):
        # Call each model on the input, and return each output
        actor_logits = self.actor_model(inputs)
        value = self.critic_model(inputs)

        return actor_logits, value

    def get_action_value(self, obs):
        '''
        Samples action and returns scalar values instead of tensors
        '''
        # Call each model on the input
        logits, value  = self.predict(obs)

        # Sample from the actor model distribution, then convert to scalar values
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=-1).numpy()
        action = action[0]
        value = value[0][0]

        return action, value

class MasterAgent():
    '''
    Master Agent
    - Builds and runs our async workers, and maintains our global model
    Arguments:
    num_episodes: Total number of episodes to be played by workers
    num_actions: Number of possible actions in the environment
    state_size: Expected size of state returned from environment
    stack_size: Number of stacked frames our model will consider
    actor_fc: Iterable containing the amount of neurons per layer for the actor model
    critic_fc: Iterable containing the amount of neurons per layer for the critic model
        ex: (1024, 512, 256) would make 3 fully connected layers, with 1024, 512 and 256
            layers respectively
    learning_rate: Learning rate
    env_func: Callable function that builds an environment for each worker. Will be passed an idx
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
    checkpoint_period: Numer of epochs to wait before saving a model
    memory_period: Number of epochs to wait before saving a memory
    '''
    def __init__(self,
                 num_episodes=1000,
                 num_actions=3,
                 state_size=[4],
                 stack_size=10,
                 actor_fc=(512,256),
                 critic_fc=(512,256),
                 learning_rate=0.00042,
                 env_func=None,
                 gamma=0.99,
                 entropy_discount=0.05,
                 value_discount=0.1,
                 boredom_thresh=10,
                 update_freq=650,
                 summary_writer=None,
                 load_path=None,
                 memory_path=None,
                 save_path="/tmp/a3c",
                 log_period=10,
                 checkpoint_period=10,
                 memory_period=None):

        self.num_episodes = num_episodes
        self.num_actions = num_actions
        self.state_size = state_size
        self.stack_size = stack_size
        self.update_freq = update_freq
        self.actor_fc = actor_fc
        self.critic_fc = critic_fc

        # Build optimizer and environment
        self.opt = tf.keras.optimizers.Adam(learning_rate)
        self.env_func = env_func

        self.gamma = gamma
        self.entropy_discount = entropy_discount
        self.value_discount = value_discount
        self.boredom_thresh = boredom_thresh
        self.update_freq = update_freq

        # Build the global model
        self.global_model = ActorCriticModel(num_actions=self.num_actions,
                                             state_size=self.state_size,
                                             stack_size=self.stack_size,
                                             actor_fc=self.actor_fc,
                                             critic_fc=self.critic_fc)
        # Load weights into global model only if a path is defined
        if load_path != None:
            self.global_model.load_weights(load_path)
            print("Loaded model from {}".format(load_path))

        self.summary_writer = summary_writer
        self.memory_path = memory_path
        self.save_path = save_path
        self.log_period = log_period
        self.checkpoint_period = checkpoint_period
        self.memory_period = memory_period
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def distributed_train(self):
        '''
        Builds, starts and ends the multithread agents
        '''
        # Build the queue that allows our workers to communicate with our main thread
        res_queue = Queue()

        # Build each of our multithread workers
        workers = []
        for i in range(multiprocessing.cpu_count()):
            # Build worker's local model
            worker_model = ActorCriticModel(num_actions=self.num_actions,
                                            state_size=self.state_size,
                                            stack_size=self.stack_size,
                                            actor_fc=self.actor_fc,
                                            critic_fc=self.critic_fc)
            # Build the worker
            workers.append(Worker(idx=i,
                           global_model=self.global_model,
                           local_model=worker_model,
                           result_queue=res_queue,
                           max_episodes=self.num_episodes,
                           num_actions=self.num_actions,
                           state_size=self.state_size,
                           stack_size=self.stack_size,
                           opt=self.opt,
                           env_func=self.env_func,
                           gamma=self.gamma,
                           entropy_discount=self.entropy_discount,
                           value_discount=self.value_discount,
                           boredom_thresh=self.boredom_thresh,
                           update_freq=self.update_freq,
                           summary_writer=self.summary_writer,
                           memory_path=self.memory_path,
                           save_path=self.save_path,
                           log_period=self.log_period,
                           checkpoint_period=self.checkpoint_period,
                           memory_period=self.memory_period))
            print("Worker {} built!".format(i))

        # Start each of our workers
        for i, worker in enumerate(workers):
            print("Starting worker {}".format(i))
            worker.start()

        # Keep track of the returns from our workers
        # Stops all workers after one returns None
        # Worker only returns None when requisite episodes are played
        moving_average_rewards = []
        while True:
            reward = res_queue.get()
            if reward is not None:
                moving_average_rewards.append(reward)
            else:
                break
        [w.join() for w in workers]
        print("Done!")
        return moving_average_rewards

    def human_train(self, data_path, train_steps, batch_size):
        '''
        Trains our actor model on human data
        Arguments:
        data_path: path to the human data stored as a pickle file of the Memory object
        train_steps: Number of epochs to train the classifier
        batch_size: Batch size of the classifier
        '''
        # Load human input from pickle file
        data_file = open(data_path, 'rb')
        memory_list = pickle.load(data_file)
        data_file.close()
        print("Loaded files")

        # Count the actions in the dataset, then create a list of inverse ratios
        # for weighting our loss. This lets us combat unbalanced data
        counts = np.zeros(self.num_actions)
        for memory in memory_list:
            for action in memory.actions:
                if action >= self.num_actions: continue
                counts[action] += 1
        print("Count of actions: {}".format(counts))
        counts = [(sum(counts) - c) / sum(counts) for c in counts]

        # A generator that iterates over our dataset and returns one action / stacked state pair
        def gen():
            while True:
                for memory in memory_list:
                    prev_states = [np.random.random(tuple(self.state_size)) for _ in range(self.stack_size)]
                    for index, (action, state) in enumerate(zip(memory.actions, memory.states)):
                        if action >= self.num_actions: continue
                        prev_states = prev_states[1:] + [state]
                        stacked_state = np.concatenate(prev_states, axis=-1)
                        yield (action, stacked_state)

        # Build a dataset from the generator, shuffle and batch it then turn it into an iterator
        dataset = tf.data.Dataset.from_generator(generator=gen,
                                                 output_types=(tf.float32, tf.float32))
        dataset = dataset.shuffle(10000).batch(batch_size)
        generator = iter(dataset)

        print("Starting steps...")
        for train_step in range(train_steps):
            # Get our batch of actions and states from the dataset
            actions, states  = next(generator)

            # Build a batch of weights corrosoponding to our actions
            weights = [counts[action] for action in actions]

            # Compute loss
            with tf.GradientTape() as tape:
                total_loss = self.compute_loss(actions,
                                               states,
                                               weights)

            # Calculate and apply gradients
            total_grads = tape.gradient(total_loss, self.global_model.actor_model.trainable_weights)
            self.opt.apply_gradients(zip(total_grads, self.global_model.actor_model.trainable_weights))
            print("Step: {} | Loss: {}".format(train_step, total_loss))

            # Every 100 steps, run validation
            if train_step % 100 == 0:
                # Get our batch of actions and states from the dataset
                target_actions, states  = next(generator)

                # Get actor logits across our batch, then argmax to get our final answers
                predict_actions = self.global_model.actor_model(states)
                predict_actions = [np.argmax(distribution) for distribution in predict_actions]

                # Compare our final answers to the labels
                target_actions = list(target_actions.numpy())
                correct = [1 if t == p else 0 for t, p in zip(target_actions, predict_actions)]

                print("Accuracy: {}".format(sum(correct) / len(correct)))


        # Initialize the critic model to predict 0 at the start - minimium expectations
        critic_batch_size = 100
        critic_steps = 1000
        self.initialize_critic_model(critic_batch_size, critic_steps)

        # Save the human data trained model
        _save_path = os.path.join(self.save_path, "human_trained_model.h5")
        os.makedirs(os.path.dirname(_save_path), exist_ok=True)
        self.global_model.save_weights(_save_path)
        print("Classifier saved to {}".format(_save_path))

        # Fixes a TF 2.0 bug with async and datasets. Don't ask
        gc.collect()

    def compute_loss(self,
                     actions,
                     states,
                     weights):
        '''
        Computes loss for training as a classifier
        Arguments:
        actions: Batch of correct actions taken from human daat
        states: Batch of stacked states
        weights: Batch of weights corrosponding to the actions
        '''
        # Get logits
        logits = self.global_model.actor_model(states)

        # Calculate weighted sparse crossentropy between labels and our predictions
        weighted_sparse_crossentropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        policy_loss = weighted_sparse_crossentropy(actions[:, None], logits, sample_weight=weights)

        return policy_loss

    def initialize_critic_model(self, batch_size, critic_steps):
        '''
        Initializes the critic model to 0 expctations
        Arguments:
        batch_size: Number of random states to initialize to 0
        critic_steps: Number of times we push towards 0
        '''
        # Generate the batch of 0 rewards
        zero_rewards = np.zeros(batch_size)
        for critic_step in range(critic_steps):
            # Generate the random states
            random_states = np.random.random((batch_size,) + tuple(self.state_size[:-1] + [self.state_size[-1] * self.stack_size]))

            # Push to 0
            with tf.GradientTape() as tape:
                values = self.global_model.critic_model(random_states)
                value_loss = keras.losses.mean_squared_error(zero_rewards[:, None], values)

            value_grads = tape.gradient(value_loss, self.global_model.critic_model.trainable_weights)
            self.opt.apply_gradients(zip(value_grads, self.global_model.critic_model.trainable_weights))

    def play(self, env):
        '''
        Watch our model play the game!
        Arguments:
        env: The environment to play on
        '''
        # Set initial states
        state, observation = env.reset()
        done = False
        step_counter = 0
        reward_sum = 0
        rolling_average_state = np.zeros(state.shape) + (0.2 * state)
        memory = Memory()

        try:
            # Initialize our stack memory with random values
            prev_states = [np.random.random(state.shape) for _ in range(self.stack_size)]
            
            # Play an episode!
            while not done:
                # Update our stack memory with our current state
                prev_states = prev_states[1:] + [state]

                # Calculate the deviation between the rolling average of previous frames and the current frame
                # Allows us to detect is we aren't moving, and take action
                _deviation = tf.reduce_sum(tf.math.squared_difference(rolling_average_state, state))
                if step_counter > 10 and _deviation < self.boredom_thresh:
                    # Take an action that isn't the previous action
                    possible_actions = np.delete(np.array(range(self.num_actions)), action)
                    action = np.random.choice(possible_actions)
                else:
                    # Convert our stack memory into a single vector
                    stacked_state = np.concatenate(prev_states, axis=-1)

                    # Calculate logits, then argmax to get our most confident action
                    logits = self.global_model.actor_model(stacked_state[None, :])
                    distribution = tf.squeeze(tf.nn.softmax(logits)).numpy()
                    action = np.argmax(logits)

                # Take our step
                (new_state, reward, done, _), new_observation = env.step(action)

                # Save data to memory. Only save extra metrics if saving memory in the end
                memory.store(state, action, reward)
                if self.memory_path:
                    memory.obs.append(observation)
                    memory.probs.append(distribution)
                    value = self.global_model.critic_model(stacked_state[None, :])
                    memory.values.append(value)

                # Update values for next iteration
                state = new_state
                observation = new_observation
                reward_sum += reward
                rolling_average_state = rolling_average_state * 0.8 + new_state * 0.2
                print("{}. Reward: {}, action: {}".format(step_counter, reward_sum, action))
                step_counter += 1
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        finally:
            # If there is a memory path, save the memory
            if self.memory_path:
                _mem_path = os.path.join(self.memory_path, "evaluation_memory")
                os.makedirs(os.path.dirname(_mem_path), exist_ok=True)
                output_file = open(_mem_path, 'wb+')
                pickle.dump(memory, output_file)
        return floor

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
            ep_steps = 0
            self.ep_loss = 0

            action = 0
            time_count = 0
            done = False
            # Initialize our stack memory with random values
            prev_states = [np.random.random(state.shape) for _ in range(self.stack_size)]

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
                    action, _ = self.local_model.get_action_value(stacked_state[None, :])

                # Take our step
                (new_state, reward, done, _), new_obs  = self.env.step(action)

                # Save data to memory. Only save extra metrics if saving memory in the end
                mem.store(state, action, reward)
                if save_visual:
                    mem.obs.append(obs)
                ep_reward += reward

                # Calculate a gradient if we've hit update_freq many steps or have reached the end of an episode
                if time_count == self.update_freq or done:
                    # Calculate loss based on trajectory
                    with tf.GradientTape() as tape:
                        total_loss  = self.compute_loss(mem, done, self.gamma, save_visual, stacked_state)
                    self.ep_loss += total_loss

                    # Calculate gradient for local model
                    total_grads = tape.gradient(total_loss, self.local_model.trainable_weights)

                    # Apply gradient to global model
                    self.opt.apply_gradients(zip(total_grads, self.global_model.trainable_weights))

                    # Update local model with new global model weights
                    self.local_model.set_weights(self.global_model.get_weights())
                    
                    if done:
                        self.log_episode(save_visual, current_episode, ep_steps, ep_reward, mem, total_loss)
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
                     stacked_state):
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
            reward_sum = self.local_model.critic_model(stacked_state[None, :])
            reward_sum = np.squeeze(reward_sum.numpy())

        # Calculate discounted rewards
        discounted_rewards = []
        for reward in memory.rewards[::-1]:
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        # Build batch of stacked states
        stacked_states = []
        prev_states = [np.random.random(tuple(self.state_size)) for _ in range(self.stack_size)]
        for index, (state, action) in enumerate(zip(memory.states, memory.actions)):
            prev_states = prev_states[1:] + [state]
            stacked_states.append(np.concatenate(prev_states, axis=-1))

        # Get batch of logits and values
        logits, values = self.local_model(np.array(stacked_states))

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
               (entropy_loss * self.entropy_discount)

    def log_episode(self, save_visual, current_episode, ep_steps, ep_reward, mem, total_loss):
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
        Worker.global_moving_average_reward = \
            record(Worker.global_episode, ep_reward, self.worker_idx,
                   Worker.global_moving_average_reward, self.result_queue,
                   self.ep_loss, ep_steps)

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
                tf.summary.scalar("Episode Loss", tf.reduce_sum(total_loss), current_episode)
                tf.summary.scalar("Moving Global Average", Worker.global_moving_average_reward, current_episode)
        if current_episode % self.checkpoint_period == 0:
            _save_path = os.path.join(self.save_path, "worker_{}_model_{}.h5".format(self.worker_idx, current_episode))
            self.local_model.save_weights(_save_path)
            print("Checkpoint saved to {}".format(_save_path))
