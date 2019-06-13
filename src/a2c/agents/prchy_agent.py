
import os
from tqdm import tqdm
from collections import deque

import numpy as np
import tensorflow as tf

from src.a2c.envs.parallel_env import ParallelEnv
from src.a2c.models.lstm_actor_critic_model import LSTMActorCriticModel
from src.a2c.models.actor_critic_model import ActorCriticModel
from src.a2c.runners.lstm_runner import LSTMRunner
from src.a2c.agents.lstm_agent import LSTMAgent

class PrierarchyAgent(LSTMAgent):
    '''
    PPO Agent class. Trains the model

    train_steps: Number of episodes to play and train on
    update_epochs: Number of update epochs to run per train step
    num_minibatches: Number of batches to split one train step batch into
    learning_rate: Learning_rate
    entropy_discount: Amount to discount entropy loss by relative to policy loss
    value_discount: Amount to discount value loss by relative to policy loss
    epsilon: Amount to clip action probability by
    num_steps: Number of steps for each environment to take per rollout
    num_envs: Number of environments to run in parallel
    num_actions: Number of actions for model to output
    actor_fc: Actor model dense layers topology
    critic_fc: Critic model dense layers topology
    conv_size: Conv model topology
    '''
    def __init__(self,
                 train_steps=None,
                 update_epochs=None,
                 num_minibatches=None,
                 learning_rate=0.00042,
                 kl_discount=0.01,
                 value_discount=0.5,
                 epsilon=0.2,
                 num_steps=None,
                 env=None,
                 num_envs=None,
                 num_actions=None,
                 actor_fc=None,
                 critic_fc=None,
                 conv_size=None,
                 before_fc=None,
                 lstm_size=None,
                 gae=False,
                 retro=False,
                 logging_period=25,
                 checkpoint_period=50,
                 output_dir="/tmp/a2c",
                 restore_dir=None,
                 restore_cnn_dir=None,
                 prior_dir=None,
                 wandb=None,
                 build=True):

        super().__init__(train_steps=train_steps,
                         update_epochs=update_epochs,
                         num_minibatches=num_minibatches,
                         learning_rate=learning_rate,
                         value_discount=value_discount,
                         epsilon=epsilon,
                         num_steps=num_steps,
                         env=env,
                         num_envs=num_envs,
                         num_actions=num_actions,
                         actor_fc=actor_fc,
                         critic_fc=critic_fc,
                         conv_size=conv_size,
                         before_fc=before_fc,
                         lstm_size=lstm_size,
                         gae=gae,
                         retro=retro,
                         logging_period=logging_period,
                         checkpoint_period=checkpoint_period,
                         output_dir=output_dir,
                         restore_dir=restore_dir,
                         restore_cnn_dir=restore_cnn_dir,
                         wandb=wandb,
                         build=False)
        self.kl_discount = kl_discount
        self.small_value = 0.0000001
        self.prior_states = np.zeros((env.num_envs, lstm_size * 2)).astype(np.float32)
        if build:
            if prior_dir == None:
                raise ValueError('Weights for the prior must be specified')

            self.prior = LSTMActorCriticModel(num_actions=num_actions,
                                              state_size=self.env.state_size,
                                              stack_size=self.env.stack_size,
                                              num_steps=num_steps,
                                              num_envs=self.env.num_envs,
                                              actor_fc=actor_fc,
                                              critic_fc=critic_fc,
                                              conv_size=conv_size,
                                              before_fc=before_fc,
                                              lstm_size=lstm_size,
                                              retro=retro)
            self.prior.load_weights(prior_dir)

            # Build model
            self.model = LSTMActorCriticModel(num_actions=num_actions,
                                              state_size=self.env.state_size,
                                              stack_size=self.env.stack_size,
                                              num_steps=num_steps,
                                              num_envs=self.env.num_envs,
                                              actor_fc=actor_fc,
                                              critic_fc=critic_fc,
                                              conv_size=conv_size,
                                              before_fc=before_fc,
                                              lstm_size=lstm_size,
                                              retro=retro)
            self.model.load_weights(prior_dir)

                                          
            # Build runner
            self.runner = LSTMRunner(env=self.env,
                                     model=self.model,
                                     num_steps=num_steps,
                                     gamma=0.9999)

    def train(self):
        for i in range(self.train_steps):
            print("\nStarting training step {}...\n".format(i))

            # Reset step logging
            step_policy_loss, step_kl_loss, step_value_loss, step_clip_frac = [], [], [], []
            # Generate indicies for minibatch sampling
            assert self.num_envs % self.num_minibatches == 0
            env_indicies = np.arange(self.num_envs)
            flat_indicies = np.arange(self.b_size).reshape(self.num_envs, self.num_steps)
            new_prior_states = np.zeros((self.env.num_envs, self.model.lstm_size * 2)).astype(np.float32)
            # Generate a batch from one rollout
            b_obs, b_rewards, b_dones, b_actions, b_values, b_probs, b_states, true_reward, ep_infos = self.runner.generate_batch()
            
            # PPO Updates!
            print("\nStarting PPO updates...")
            for e in range(self.update_epochs):
                # Reset epoch logging
                epoch_policy_loss, epoch_kl_loss, epoch_value_loss, epoch_clip_frac = [], [], [], []
                # Shuffle indicies
                np.random.shuffle(env_indicies)

                for start in tqdm(range(0, self.num_envs, self.envs_per_batch), "   Epoch {}".format(e)):
                    # Generate minibatch
                    end = start + self.envs_per_batch
                    mb_env_inds = env_indicies[start:end]
                    mb_flat_inds = flat_indicies[mb_env_inds].ravel()
                    mb_obs, mb_rewards, mb_dones, mb_actions, mb_values, mb_probs = \
                            (arr[mb_flat_inds] for arr in (b_obs, b_rewards, b_dones,
                                                          b_actions, b_values, b_probs))
                    mb_states = b_states[mb_env_inds]
                    mb_prior_states = self.prior_states[mb_env_inds]
                    # Calculate advantages
                    advs = mb_rewards - mb_values

                    # Calculate loss
                    with tf.GradientTape() as tape:
                        # Get actions probabilities and values for all states
                        mb_obs = self.model.process_inputs(mb_obs)
                        logits, values, _ = self.model([mb_obs, mb_states, mb_dones])

                        # Model returns un-softmaxed logits
                        logits = tf.nn.softmax(logits)

                        # Get action probabilities
                        one_hot_actions = tf.one_hot(mb_actions, self.num_actions)
                        cur_probs = tf.reduce_sum(logits * one_hot_actions, axis=-1)

                        # Calculate our policy loss
                        ratio = cur_probs / mb_probs
                        unclipped_policy_loss = -advs * ratio
                        clipped_policy_loss = -advs * tf.clip_by_value(ratio,
                                                                      1 - self.epsilon,
                                                                      1 + self.epsilon) 
                        policy_loss = tf.reduce_mean(tf.maximum(unclipped_policy_loss,
                                                                clipped_policy_loss))

                        # Calculate our KL divergence relative to the prior
                        prior_logits, _, mb_new_prior_states = self.prior([mb_obs, mb_prior_states, mb_dones])
                        if end == self.num_envs:
                            new_prior_states[mb_env_inds] = mb_new_prior_states.numpy()
                        prior_logits = tf.nn.softmax(prior_logits)
                        kl_loss = tf.reduce_mean(prior_logits *
                                                 tf.math.log(prior_logits / (logits + self.small_value) + self.small_value))

                        # Calculate our value loss
                        mse = tf.keras.losses.MeanSquaredError()
                        value_loss = mse(mb_rewards[:, None], values)

                        total_loss = policy_loss + \
                                     (value_loss * self.value_discount) + \
                                     -(kl_loss * self.kl_discount)

                    # Calculate and apply gradient
                    total_grads = tape.gradient(total_loss, self.model.trainable_weights)
                    total_grads, total_grad_norm = tf.clip_by_global_norm(total_grads, 0.5)
                    self.opt.apply_gradients(zip(total_grads, self.model.trainable_weights))

                    # Store minibatch metrics
                    epoch_policy_loss.append(policy_loss.numpy())
                    epoch_kl_loss.append(kl_loss.numpy())
                    epoch_value_loss.append(value_loss.numpy())

                    # Calculate the fraction of our mini batch that was clipped
                    clips = tf.greater(tf.abs(ratio - 1), self.epsilon)
                    clip_frac = tf.reduce_mean(tf.cast(clips, tf.float32))
                    epoch_clip_frac.append(clip_frac)

                # Log epoch data
                self.log_epoch(e, epoch_policy_loss, epoch_kl_loss, epoch_value_loss, epoch_clip_frac)
                # Store epoch metrics
                step_policy_loss.append(np.mean(epoch_policy_loss))
                step_kl_loss.append(np.mean(epoch_kl_loss))
                step_value_loss.append(np.mean(epoch_value_loss))

            # Log step data
            self.log_step(b_rewards, b_values, b_probs, step_policy_loss,
                          step_kl_loss, step_value_loss, epoch_clip_frac,
                          ep_infos, i)
            self.prior_states = new_prior_states

    def log_epoch(self, e, policy_loss, kl_loss, value_loss, clip_frac):
        avg_policy_loss = np.mean(policy_loss)
        avg_kl_loss = np.mean(kl_loss)
        avg_value_loss = np.mean(value_loss)
        avg_clip_frac = np.mean(clip_frac)

        print("\t\t| Policy Loss: {} | KL Loss: {} | Value Loss: {} |".format(avg_policy_loss, avg_kl_loss, avg_value_loss))
        print("\t\t| Fraction Clipped: {} |".format(avg_clip_frac))

    def log_step(self, rewards, values, probs, policy_loss, kl_loss, value_loss, clip_frac, ep_infos, i):
        # Pull specific info from info array and store in queue
        for info in ep_infos:
            self.floor_queue.append(info["floor"])
            self.reward_queue.append(info["total_reward"])
            self.episodes += 1
        
        avg_floor = 0 if len(self.floor_queue) == 0 else sum(self.floor_queue) / len(self.floor_queue)
        avg_reward = 0 if len(self.reward_queue) == 0 else sum(self.reward_queue) / len(self.reward_queue)
        avg_policy_loss = np.mean(policy_loss)
        avg_kl_loss = np.mean(kl_loss)
        avg_value_loss = np.mean(value_loss)
        avg_clip_frac = np.mean(clip_frac)
        explained_variance, env_variance = self.explained_variance(values, rewards)

        print("\nTrain Step Metrics:")
        print("\t| Total Episodes: {} | Average Floor: {} | Average Reward: {} |".format(self.episodes, avg_floor, avg_reward))
        print("\t| Policy Loss: {} | KL Loss: {} | Value Loss: {} |".format(avg_policy_loss, avg_kl_loss, avg_value_loss))
        print("\t| Explained Variance: {} | Environment Variance: {} |".format(explained_variance, env_variance))
        print()

        # Periodically log
        if i % self.logging_period == 0:
            with self.summary_writer.as_default():
                tf.summary.scalar("Average Floor", avg_floor, i)
                tf.summary.scalar("Average Reward", avg_reward, i)
                tf.summary.scalar("Policy Loss", avg_policy_loss, i)
                tf.summary.scalar("Entropy Loss", avg_kl_loss, i)
                tf.summary.scalar("Value Loss", avg_value_loss, i)
                tf.summary.scalar("Explained Variance", explained_variance, i)
                tf.summary.scalar("Fraction Clipped", avg_clip_frac, i)
            if self.wandb != None:
                self.wandb.log({"epoch": i,
                                "Average Floor": avg_floor,
                                "Average Reward": avg_reward,
                                "Average Floor Distribution": self.wandb.Histogram(self.floor_queue, num_bins=25),
                                "Policy Loss": avg_policy_loss,
                                "Entropy Loss": avg_kl_loss,
                                "Value Loss": avg_value_loss,
                                "Explained Variance": explained_variance,
                                "Fraction Clipped": avg_clip_frac,
                                "Probabilities": self.wandb.Histogram(probs, num_bins=10)})
        # Periodically save checkoints
        if i % self.checkpoint_period == 0:
            model_save_path = os.path.join(self.checkpoint_dir, "model_{}.h5".format(i))
            self.model.save_weights(model_save_path)
            print("Model saved to {}".format(model_save_path))
