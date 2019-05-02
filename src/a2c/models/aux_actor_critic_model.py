
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from src.a2c.models.actor_critic_model import ActorCriticModel

class AuxActorCriticModel(ActorCriticModel):
    '''
    Actor Critic Model
    - Contains both our actor and critic model seperately
    as Keras Models
    Arguments:
    num_actions: Number of logits the actor model will output
    state_size: List containing the expected size of the state
    max_floor: Maximium number of floors reachable
    stack_size: Numer of states we can expect in our stack
    actor_fc: Iterable containing the amount of neurons per layer for the actor model
    critic_fc: Iterable containing the amount of neurons per layer for the critic model
        ex: (1024, 512, 256) would make 3 fully connected layers, with 1024, 512 and 256
            layers respectively
    conv_size: Iterable containing the kernel size, stride and number of filters for each
               convolutional layer.
        ex: ((8, 4, 16)) would make 1 convolution layer with an 8x8 kernel, (4, 4) stride
            and 16 filters
    max_pooling: Whether or not to throw max pooling layers after convolutional layers.
                 Currently only does (2,2) pooling.
    '''
    def __init__(self,
                 num_actions=None,
                 num_aux=9,
                 state_size=None,
                 max_floor=25,
                 stack_size=None,
                 actor_fc=None,
                 critic_fc=None,
                 conv_size=None,
                 retro=False,
                 build=True):
        super().__init__(num_actions=num_actions,
                         state_size=state_size,
                         max_floor=max_floor,
                         stack_size=stack_size,
                         actor_fc=actor_fc,
                         critic_fc=critic_fc,
                         conv_size=conv_size,
                         retro=retro,
                         build=False)

        self.aux = layers.Dense(num_aux, name="aux", activation="softmax")

        # Take a step with random input to build the model
        if build and retro:
            self.step([[np.random.random((tuple(self.state_size))).astype(np.float32),
                        np.random.random((max_floor + 1,)).astype(np.float32)]])
        elif build:
            self.step([[np.random.random((tuple(self.state_size))).astype(np.float32),
                        np.random.random((max_floor + 1,)).astype(np.float32),
                        np.random.random((2,)).astype(np.float32)]])

    def call(self, inputs):
        if self.retro:
            conv_input, critic_input = inputs
        else:
            conv_input, critic_input, shared_input = inputs

        # Run convs on conv input
        if self.convs is not None:
            conv_out = self.convs(conv_input)
            conv_out = self.flatten(conv_out)
            shared_dense = conv_out
        else:
            shared_dense = conv_input

        # Add shared information
        if self.retro is False:
            shared_dense = layers.concatenate([shared_dense, shared_input])

        # Run actor layers
        actor_dense = shared_dense
        for l in self.actor_fc:
            actor_dense = l(actor_dense)
        actor_logits = self.actor_logits(actor_dense)

        # Run critic layers
        critic_dense = layers.concatenate([shared_dense, critic_input])
        for l in self.critic_fc:
            critic_dense = l(critic_dense)
        value = self.value(critic_dense)

        # Run aux layers
        aux = self.aux(conv_out)

        return actor_logits, value, aux

    def step(self, inputs):
        inputs = self.process_inputs(inputs)

        # Make predictions on input
        logits, values, _ = self.predict(inputs)
        probs = tf.nn.softmax(logits)

        # Sample from probability distributions
        actions = tf.squeeze(tf.random.categorical(logits, 1), axis=-1).numpy()

        # Get action probabilities
        one_hot_actions = tf.one_hot(actions, self.num_actions)
        action_probs = tf.reduce_sum(probs * one_hot_actions, axis=-1).numpy()

        # TODO Fix bug where this line breaks the program when there is only 1 env
        values = np.squeeze(values)

        return actions, values, action_probs

    def get_values(self, inputs):
        inputs = self.process_inputs(inputs)

        _, values, _ = self.predict(inputs)
        values = np.squeeze(values)

        return values

    def process_inputs(self, inputs):
        # Convert n_envs x n_inputs list to n_inputs x n_envs list if we have
        # multiple inputs
        inputs = [np.asarray(l) for l in zip(*inputs)]
        return inputs




if __name__ == '__main__':
    model = AuxActorCriticModel(num_actions=4,
                                state_size=[84,84,1],
                                stack_size=4,
                                actor_fc=[32,16],
                                critic_fc=[32,16],
                                retro=True,
                                conv_size="quake")
    print(model.trainable_weights)
    input()

    ret = model.step(np.stack([[np.random.random((84, 84, 4)).astype(np.float32),
                np.random.random((26,)).astype(np.float32)] for _ in range(5)]))
    print(ret)
    print(ret[0].shape)
    print(type(ret[0]))
    print(ret[1].shape)
    print(type(ret[1]))
    ret2 = model.get_values(np.stack([[np.random.random((84, 84, 4)).astype(np.float32),
                np.random.random((26,)).astype(np.float32)] for _ in range(5)]))
    print((ret2,))
    print(type(ret2))
    print(ret2.shape)
