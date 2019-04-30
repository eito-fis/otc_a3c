
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class ActorCriticModel(tf.keras.models.Model):
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
    '''
    def __init__(self,
                 num_actions=None,
                 state_size=None,
                 max_floor=25,
                 stack_size=None,
                 actor_fc=None,
                 critic_fc=None,
                 conv_size=None,
                 retro=True,
                 build=True):
        super().__init__()
        self.num_actions = num_actions
        self.retro = retro

        # Multiply the final dimension of the state by stack_size to get correct input_size
        self.state_size = state_size[:-1] + [state_size[-1] * stack_size]

        # Build convolutional layers
        if conv_size is not None:
            if isinstance(conv_size, tuple):
                self.convs = Custom_Convs(conv_size)
            elif conv_size == "quake":
                self.convs = Quake_Block()
            else:
                raise ValueError("Invalid CNN Topology")
        else: self.convs = None
        self.flatten = layers.Flatten()
        
        # Build the fully connected layers for the actor and critic models
        self.actor_fc = [layers.Dense(neurons, activation="relu", name="actor_dense_{}".format(i)) for i,(neurons) in enumerate(actor_fc)]
        self.critic_fc = [layers.Dense(neurons, activation="relu", name="critic_dense_{}".format(i)) for i,(neurons) in enumerate(critic_fc)]

        # Build the output layers for the actor and critic models
        self.actor_logits = layers.Dense(num_actions, name='policy_logits')
        self.value = layers.Dense(1, name='value')

        # Build the final total model
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
            shared_dense = self.flatten(conv_out)
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
            critic__dense = l(critic_dense)
        value = self.value(critic_dense)

        return actor_logits, value

    def step(self, inputs):
        inputs = self.process_inputs(inputs)

        # Make predictions on input
        logits, values = self.predict(inputs)
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

        _, values = self.predict(inputs)
        values = np.squeeze(values)

        return values

    def process_inputs(self, inputs):
        # Convert n_envs x n_inputs list to n_inputs x n_envs list if we have
        # multiple inputs
        inputs = [np.asarray(l) for l in zip(*inputs)]
        return inputs

class Custom_Convs(tf.keras.Model):
    def __init__(self, conv_size, actv="relu"):
        super().__init__(name='')

        self.convs = [layers.Conv2D(padding="same",
                                    kernel_size=k,
                                    strides=s,
                                    filters=f,
                                    activation=actv,
                                    name="conv_{}".format(i))
                      for i,(k,s,f) in enumerate(conv_size)]
    
    def call(self, x):
        for conv in self.convs:
            x = conv(x)
        return x

class Quake_Block(tf.keras.Model):
    def __init__(self):
        super().__init__(name='')

        # Quake 3 Deepmind style convolutions
        # Like dopamine but with an additional 3x3 kernel, and skip connections
        self.conv2A = layers.Conv2D(padding="same", kernel_size=8, strides=4, filters=32, activation="relu")
        self.conv2B = layers.Conv2D(padding="same", kernel_size=4, strides=2, filters=64, activation="relu")
        self.conv2C = layers.Conv2D(padding="same", kernel_size=3, strides=1, filters=64)
        self.activationC = layers.Activation("relu")
        self.conv2D = layers.Conv2D(padding="same", kernel_size=3, strides=1, filters=64)
        self.activationD = layers.Activation("relu")

    def call(self, x):
        x = self.conv2A(x)
        x = skip_1 = self.conv2B(x)

        x = self.conv2C(x)
        x = skip_2 = layers.add([x, skip_1])
        x = self.activationC(x)

        x = self.conv2D(x)
        x = layers.add([x, skip_2])
        x = self.activationD(x)

        return x



if __name__ == '__main__':
    model = ActorCriticModel(num_actions=4,
                             state_size=[84,84,1],
                             stack_size=4,
                             actor_fc=[32,16],
                             critic_fc=[32,16],
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
