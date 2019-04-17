
import numpy as np
import tensorflow as tf

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
    max_pooling: Whether or not to throw max pooling layers after convolutional layers.
                 Currently only does (2,2) pooling.
    '''
    def __init__(self,
                 num_actions=None,
                 state_size=None,
                 max_floor=25,
                 stack_size=None,
                 actor_fc=None,
                 critic_fc=None,
                 conv_size=None,
                 max_pooling=False):
        super().__init__()

        # Multiply the final dimension of the state by stack_size to get correct input_size
        self.state_size = state_size[:-1] + [state_size[-1] * stack_size]

        # Build general input and any additional input specific to models
        model_input = tf.keras.layers.Input(shape=tuple(self.state_size))

        # Build the fully connected layers for shared convolutional layers
        if conv_size is not None:
            conv_x = model_input
            for i,(k,s,f) in enumerate(conv_size):
                conv_x = tf.keras.layers.Conv2D(padding="same",
                                                kernel_size=k,
                                                strides=s,
                                                filters=f,
                                                use_bias=False,
                                                name="conv_{}".format(i))(conv_x)
                conv_x = tf.keras.layers.BatchNormalization(name="batch_norm_{}".format(i))(conv_x)
                conv_x = tf.keras.layers.Activation("relu", name="conv_activation_{}".format(i))(conv_x)
                if max_pooling:
                    conv_x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(conv_x)
            flatten = tf.keras.layers.Flatten(name="Flatten")(conv_x)
            actor_x = flatten
        else:
            actor_x = model_input

        # Build the fully connected layers for the actor and critic models
        for i,(neurons) in enumerate(actor_fc):
            actor_x = tf.keras.layers.Dense(neurons,
                                            activation="relu",
                                            name="dense_{}".format(i))(actor_x)

        # Build the output layers for the actor and critic models
        actor_logits = tf.keras.layers.Dense(num_actions, name='policy_logits')(actor_x)
        value = tf.keras.layers.Dense(1, name='value')(actor_x)

        # Build the final total model
        self.model = tf.keras.models.Model(inputs=[model_input],
                                           outputs=[actor_logits, value])

        # Take a step with random input to build the model
        self.step([[np.random.random((tuple(self.state_size))).astype(np.float32)]])

    def call(self, inputs):
        actor_logits, value = self.model(inputs)
        return actor_logits, value

    def step(self, inputs):
        inputs = self.process_inputs(inputs)

        # Make predictions on input
        logits, values = self.predict(inputs)

        # Sample from probability distributions
        actions = tf.squeeze(tf.random.categorical(logits, 1), axis=-1).numpy()
        probs = tf.nn.softmax(logits)
        action_probs = np.array([p[a] for p,a in zip(probs, actions)])

        # TODO Fix bug where this line breaks the program when there is only 1 env
        values = np.squeeze(values)

        return actions, values, action_probs

    def get_values(self, inputs):
        inputs = self.process_inputs(inputs)

        _, values = self.model.predict(inputs)
        values = np.squeeze(values)

        return values

    def process_inputs(self, inputs):
        # Convert n_envs x n_inputs list to n_inputs x n_envs list if we have
        # multiple inputs
        inputs = [np.asarray(l) for l in zip(*inputs)]
        return inputs

if __name__ == '__main__':
    model = ActorCriticModel(num_actions=4,
                             state_size=[84,84,1],
                             stack_size=4,
                             actor_fc=[32],
                             critic_fc=[32],
                             conv_size=((3,1,64),))
    ret = model.step(np.stack([[np.random.random((84, 84, 4)).astype(np.float32),
                np.random.random((1)).astype(np.float32)] for _ in range(5)]))
    print(ret)
    print(ret[0].shape)
    print(type(ret[0]))
    print(ret[1].shape)
    print(type(ret[1]))
    ret2 = model.get_values(np.stack([[np.random.random((84, 84, 4)).astype(np.float32),
                np.random.random((1)).astype(np.float32)] for _ in range(5)]))
    print((ret2,))
    print(type(ret2))
    print(ret2.shape)
