
import os
from tqdm import tqdm
from collections import deque
from PIL import Image

import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub

from src.a2c.envs.parallel_env import ParallelEnv
from src.a2c.models.aux_actor_critic_model import AuxActorCriticModel
from src.a2c.runners.aux_runner import AuxRunner
from src.a2c.agents.ppo_agent import PPOAgent

class AuxAgent(PPOAgent):
    '''
    Auxillary Output PPO Agent class. Trains the model

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
                 entropy_discount=0.01,
                 value_discount=0.5,
                 aux_discount=1,
                 epsilon=0.2,
                 num_steps=None,
                 env_func=None,
                 num_envs=None,
                 num_actions=None,
                 actor_fc=None,
                 critic_fc=None,
                 conv_size=None,
                 gae=False,
                 retro=False,
                 num_aux=8,
                 logging_period=25,
                 checkpoint_period=50,
                 output_dir="/tmp/a2c",
                 restore_dir=None,
                 wandb=None,
                 build=True):

        super().__init__(train_steps=train_steps,
                         update_epochs=update_epochs,
                         num_minibatches=num_minibatches,
                         learning_rate=learning_rate,
                         entropy_discount=entropy_discount,
                         value_discount=value_discount,
                         epsilon=epsilon,
                         num_steps=num_steps,
                         num_envs=num_envs,
                         num_actions=num_actions,
                         logging_period=logging_period,
                         checkpoint_period=checkpoint_period,
                         output_dir=output_dir,
                         wandb=wandb,
                         build=False)

        if build:
            # Build environment
            env_func_list = [env_func for _ in range(num_envs)]
            self.env = ParallelEnv(env_func_list)

            # Build model
            self.model = AuxActorCriticModel(num_actions=num_actions,
                                             num_aux=num_aux,
                                             state_size=self.env.state_size,
                                             stack_size=self.env.stack_size,
                                             actor_fc=actor_fc,
                                             critic_fc=critic_fc,
                                             conv_size=conv_size,
                                             retro=retro)
            if restore_dir != None:
                self.model.load_weights(restore_dir)

            self.runner = AuxRunner(env=self.env,
                                    model=self.model,
                                    num_steps=num_steps)
        self.aux_discount = aux_discount

    def train(self):
        for i in range(self.train_steps):
            print("\nStarting training step {}...\n".format(i))

            # Reset step logging
            step_policy_loss, step_entropy_loss, step_value_loss, step_aux_loss, step_clip_frac = [], [], [], [], []
            # Generate indicies for minibatch sampling
            indicies = np.arange(self.b_size)
            # Generate a batch from one rollout
            b_states, b_rewards, b_dones, b_actions, b_values, b_probs, b_aux, true_reward, ep_infos = self.runner.generate_batch()
            
            # PPO Updates!
            print("\nStarting PPO updates...")
            for e in range(self.update_epochs):
                # Reset epoch logging
                epoch_policy_loss, epoch_entropy_loss, epoch_value_loss, epoch_aux_loss, epoch_clip_frac = [], [], [], [], []
                # Shuffle indicies
                np.random.shuffle(indicies)

                for start in tqdm(range(0, self.b_size, self.mb_size), "   Epoch {}".format(e)):
                    # Generate minibatch
                    end = start + self.mb_size
                    mb_indicies = indicies[start:end]
                    mb_states, mb_rewards, mb_actions, mb_values, mb_aux, mb_probs = \
                            (arr[mb_indicies] for arr in (b_states, b_rewards,
                                                          b_actions, b_values,
                                                          b_aux, b_probs))
                    # Calculate advantages
                    advs = mb_rewards - mb_values
                    # Normalize if using GAE
                    if self.gae: advs = (advs - advs.mean()) / (advs.std() + 1e-8)

                    # Calculate loss
                    with tf.GradientTape() as tape:
                        # Get actions probabilities and values for all states
                        processed_states = self.model.process_inputs(mb_states)
                        logits, values, aux_pred = self.model(processed_states)

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
                        # Calculate our entropy loss
                        cce = tf.keras.losses.CategoricalCrossentropy()
                        entropy_loss = cce(logits, logits) * -1
                        # Calculate our value loss
                        mse = tf.keras.losses.MeanSquaredError()
                        value_loss = mse(mb_rewards[:, None], values)
                        # Calculate our aux loss
                        aux_loss = cce(mb_aux, aux_pred)


                        total_loss = policy_loss + \
                                     (entropy_loss * self.entropy_discount) + \
                                     (value_loss * self.value_discount) + \
                                     (aux_loss * self.aux_discount)

                    # Calculate and apply gradient
                    total_grads = tape.gradient(total_loss, self.model.trainable_weights)
                    total_grads, total_grad_norm = tf.clip_by_global_norm(total_grads, 0.5)
                    self.opt.apply_gradients(zip(total_grads, self.model.trainable_weights))

                    # Store minibatch metrics
                    epoch_policy_loss.append(policy_loss.numpy())
                    epoch_entropy_loss.append(entropy_loss.numpy())
                    epoch_value_loss.append(value_loss.numpy())
                    epoch_aux_loss.append(aux_loss.numpy())

                    # Calculate the fraction of our mini batch that was clipped
                    clips = tf.greater(tf.abs(ratio - 1), self.epsilon)
                    clip_frac = tf.reduce_mean(tf.cast(clips, tf.float32))
                    epoch_clip_frac.append(clip_frac)

                # Log epoch data
                self.log_epoch(e, epoch_policy_loss, epoch_entropy_loss, epoch_value_loss, epoch_aux_loss, epoch_clip_frac)
                # Store epoch metrics
                step_policy_loss.append(np.mean(epoch_policy_loss))
                step_entropy_loss.append(np.mean(epoch_entropy_loss))
                step_value_loss.append(np.mean(epoch_value_loss))
                step_aux_loss.append(np.mean(epoch_aux_loss))

            # Log step data
            self.log_step(b_rewards, b_values, b_probs, step_policy_loss,
                          step_entropy_loss, step_value_loss, step_aux_loss,
                          epoch_clip_frac, ep_infos, i)

    def log_epoch(self, e, policy_loss, entropy_loss, value_loss, aux_loss, clip_frac):
        avg_policy_loss = np.mean(policy_loss)
        avg_entropy_loss = np.mean(entropy_loss)
        avg_value_loss = np.mean(value_loss)
        avg_aux_loss = np.mean(aux_loss)
        avg_clip_frac = np.mean(clip_frac)

        print("\t\t| Policy Loss: {} | Entropy Loss: {} | Value Loss: {} |".format(avg_policy_loss, avg_entropy_loss, avg_value_loss))
        print("\t\t| Aux Loss: {} |".format(avg_aux_loss))
        print("\t\t| Fraction Clipped: {} |".format(avg_clip_frac))

    def log_step(self, rewards, values, probs, policy_loss, entropy_loss, value_loss, aux_loss, clip_frac, ep_infos, i):
        # Pull specific info from info array and store in queue
        for info in ep_infos:
            self.floor_queue.append(info["floor"])
            self.reward_queue.append(info["total_reward"])
            self.episodes += 1
        
        avg_floor = 0 if len(self.floor_queue) == 0 else sum(self.floor_queue) / len(self.floor_queue)
        avg_reward = 0 if len(self.reward_queue) == 0 else sum(self.reward_queue) / len(self.reward_queue)
        avg_policy_loss = np.mean(policy_loss)
        avg_entropy_loss = np.mean(entropy_loss)
        avg_value_loss = np.mean(value_loss)
        avg_aux_loss = np.mean(aux_loss)
        avg_clip_frac = np.mean(clip_frac)
        explained_variance, env_variance = self.explained_variance(values, rewards)

        print("\nTrain Step Metrics:")
        print("\t| Total Episodes: {} | Average Floor: {} | Average Reward: {} |".format(self.episodes, avg_floor, avg_reward))
        print("\t| Policy Loss: {} | Entropy Loss: {} | Value Loss: {} |".format(avg_policy_loss, avg_entropy_loss, avg_value_loss))
        print("\t| Aux Loss: {} |".format(avg_aux_loss))
        print("\t| Explained Variance: {} | Environment Variance: {} |".format(explained_variance, env_variance))
        print()

        # Periodically log
        if i % self.logging_period == 0:
            with self.summary_writer.as_default():
                tf.summary.scalar("Average Floor", avg_floor, i)
                tf.summary.scalar("Average Reward", avg_reward, i)
                tf.summary.scalar("Policy Loss", avg_policy_loss, i)
                tf.summary.scalar("Entropy Loss", avg_entropy_loss, i)
                tf.summary.scalar("Value Loss", avg_value_loss, i)
                tf.summary.scalar("Aux Loss", avg_aux_loss, i)
                tf.summary.scalar("Explained Variance", explained_variance, i)
                tf.summary.scalar("Fraction Clipped", avg_clip_frac, i)
            if self.wandb != None:
                self.wandb.log({"epoch": i,
                                "Average Floor": avg_floor,
                                "Average Reward": avg_reward,
                                "Average Floor Distribution": self.wandb.Histogram(self.floor_queue, num_bins=25),
                                "Policy Loss": avg_policy_loss,
                                "Entropy Loss": avg_entropy_loss,
                                "Value Loss": avg_value_loss,
                                "Aux Loss": avg_aux_loss,
                                "Explained Variance": explained_variance,
                                "Fraction Clipped": avg_clip_frac,
                                "Probabilities": self.wandb.Histogram(probs, num_bins=10)})
        # Periodically save checkoints
        if i % self.checkpoint_period == 0:
            model_save_path = os.path.join(self.checkpoint_dir, "model_{}.h5".format(i))
            self.model.save_weights(model_save_path)
            print("Model saved to {}".format(model_save_path))




if __name__ == '__main__':
    from src.a2c.wrapped_obstacle_tower_env import WrappedObstacleTowerEnv
    env_filename = "../ObstacleTower/obstacletower"
    def env_func(idx):
        return WrappedObstacleTowerEnv(env_filename,
                                       worker_id=idx,
                                       gray_scale=True,
                                       realtime_mode=True)

    print("Building agent...")
    agent = PPOAgent(train_steps=1,
                     entropy_discount=0.01,
                     value_discount=0.5,
                     learning_rate=0.00000042,
                     num_steps=5,
                     env_func=env_func,
                     num_envs=4,
                     num_actions=4,
                     actor_fc=[1024,512],
                     critic_fc=[1024,512],
                     conv_size=((8,4,32), (4,2,64), (3,1,64)),
                     output_dir="./agent_test")
    print("Agent built!")

    print("Starting train...")
    agent.train()
    print("Train done!")
    