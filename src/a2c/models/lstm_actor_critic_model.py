
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class TD_Quake_Block(tf.keras.Model):
    def __init__(self):
        super(TD_Quake_Block, self).__init__(name='')

        # Quake 3 Deepmind style convolutions
        # Like dopamine but with an additional 3x3 kernel, and skip connections
        self.conv2A = layers.TimeDistributed(layers.Conv2D(padding="same", kernel_size=8, strides=4, filters=32, activation="relu"))
        self.conv2B = layers.TimeDistributed(layers.Conv2D(padding="same", kernel_size=4, strides=2, filters=64, activation="relu"))
        self.conv2C = layers.TimeDistributed(layers.Conv2D(padding="same", kernel_size=3, strides=1, filters=64))
        self.activationC = layers.Activation("relu")
        self.conv2D = layers.TimeDistributed(layers.Conv2D(padding="same", kernel_size=3, strides=1, filters=64))
        self.activationD = layers.Activation("relu")
        self.flatten = layers.TimeDistributed(layers.Flatten())

    def call(self, x):
        x = self.conv2A(x)
        x = skip_1 = self.conv2B(x)

        x = self.conv2C(x)
        x = skip_2 = layers.add([x, skip_1])
        x = self.activationC(x)

        x = self.conv2D(x)
        x = layers.add([x, skip_2])
        x = self.activationD(x)

        return (self.flatten(x))

class Custom_LSTM(layers.Layer):
    def __init__(self, lstm_size, input_size, ortho_scale):
        super(Custom_LSTM, self).__init__()
        self.lstm_size = lstm_size
        self.input_size = input_size
        self.ortho_scale = ortho_scale

    def build(self, input_shape):
        self.weight_x = self.add_weight(shape=(input_shape[-1], self.lstm_size * 4),
                                        initializer=tf.initializers.Orthogonal(gain=self.ortho_scale),
                                        trainable=True)
        self.weight_h = self.add_weight(shape=(self.lstm_size, self.lstm_size * 4),
                                        initializer=tf.initializers.Orthogonal(gain=self.ortho_scale),
                                        trainable=True)
        self.bias = self.add_weight(shape=(self.lstm_size * 4,),
                                    initializer=tf.initializers.Constant(0),
                                    trainable=True)

    def call(self, input_tensor, cell_state_hidden, mask_tensor):
        cell_state, hidden = tf.split(axis=1, num_or_size_splits=2, value=cell_state_hidden)
        print(input_tensor)
        print(cell_state)
        print(mask_tensor)
        for idx in range(self.input_size):
            _input = input_tensor[:, idx]
            mask = mask_tensor[:, idx]

            # Reset cell state and hidden if our this timestep is a restart
            cell_state = cell_state * (1 - mask)
            hidden = hidden * (1 - mask)
            
            # Run LSTM cell!
            print("=" * 20)
            print(mask)
            print(_input)
            print(self.weight_x)
            print(hidden)
            print(self.weight_h)
            input()
            gates = tf.matmul(_input, self.weight_x) + tf.matmul(hidden, self.weight_h) + self.bias
            in_gate, forget_gate, out_gate, cell_candidate = tf.split(axis=1, num_or_size_splits=4, value=gates)

            in_gate = tf.nn.sigmoid(in_gate)
            forget_gate = tf.nn.sigmoid(forget_gate)
            out_gate = tf.nn.sigmoid(out_gate)
            cell_candidate = tf.tanh(cell_candidate)

            cell_state = forget_gate * cell_state + in_gate * cell_candidate
            hidden = out_gate * tf.tanh(cell_state)

            input_tensor[idx] = hidden

        cell_state_hidden = tf.concat(axis=1, values=[cell_state, hidden])
        return input_tensor, cell_state_hidden

    def compute_output_shape(self, input_shape):
        return input_shape

class LSTMActorCriticModel(tf.keras.models.Model):
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
                 num_steps=None,
                 max_floor=25,
                 stack_size=None,
                 before_fc=None,
                 actor_fc=None,
                 critic_fc=None,
                 lstm_size=None):
        super(LSTMActorCriticModel, self).__init__()

        self.num_actions = num_actions
        self.actor_fc = actor_fc
        self.critic_fc = critic_fc
        self.lstm_size = lstm_size

        # Multiply the final dimension of the state by stack_size to get correct input_size
        self.state_size = state_size[:-1] + [state_size[-1] * stack_size]

        # Build convolutional layer
        self.convs = TD_Quake_Block()
        
        # Build fully connected layers that go between convs and LSTM
        self.before_fc = [layers.TimeDistributed(layers.Dense(neurons, activation="relu", name="actor_dense_{}".format(i))) for i,(neurons) in enumerate(before_fc)]

        # Build LSTM
        self.lstm = Custom_LSTM(lstm_size, num_steps, 1.0)

        # Build the fully connected layers for the actor and critic models
        self.actor_fc = [layers.TimeDistributed(layers.Dense(neurons, activation="relu", name="actor_dense_{}".format(i))) for i,(neurons) in enumerate(actor_fc)]
        self.critice_fc = [layers.TimeDistributed(layers.Dense(neurons, activation="relu", name="critic_dense_{}".format(i))) for i,(neurons) in enumerate(critic_fc)]

        # Build the output layers for the actor and critic models
        self.actor_logits = layers.Dense(num_actions, name='policy_logits')
        self.value = layers.Dense(1, name='value')

        # Take a step with random input to build the model
        self.step(np.random.random(((1, num_steps) + tuple(self.state_size))).astype(np.float32), np.zeros((1, lstm_size * 2)), np.zeros((1, num_steps)))

    def call(self, inputs):
        obs, cell_state_hidden, reset_mask = inputs

        x = self.convs(obs)
        for l in self.before_fc:
            x = l(x)

        x, cell_state_hidden = self.lstm(x, cell_state_hidden, reset_mask)

        actor_x = x
        for l in actor_fc:
            actor_x = l(actor_x)
        logits = self.actor_logits(actor_x)

        critic_x = x
        for l in critic_fc:
            critic_x = l(critic_x)
        value = self.value(critic_x)

        return (logits, value, cell_state_hidden)

    def step(self, inputs, cell_state_hidden, reset_mask):

        # Make predictions on input
        logits, values, cell_state_hidden = self.predict([inputs, cell_state_hidden, reset_mask])
        probs = tf.nn.softmax(logits)

        # Sample from probability distributions
        actions = tf.squeeze(tf.random.categorical(logits, 1), axis=-1).numpy()

        # Get action probabilities
        one_hot_actions = tf.one_hot(actions, self.num_actions)
        action_probs = tf.reduce_sum(probs * one_hot_actions, axis=-1).numpy()

        # TODO Fix bug where this line breaks the program when there is only 1 env
        values = np.squeeze(values)

        return actions, values, action_probs, cell_state_hidden

    def get_values(self, inputs, cell_state_hidden, reset_mask):
        _, values = self.model.predict([inputs, cell_state_hidden, reset_mask])
        values = np.squeeze(values)

        return values

    def process_inputs(self, inputs):
        # Convert n_envs x n_inputs list to n_inputs x n_envs list if we have
        # multiple inputs
        inputs = [np.asarray(l) for l in zip(*inputs)]
        return inputs




if __name__ == '__main__':
    model = LSTMActorCriticModel(num_actions=4,
                             state_size=[84,84,3],
                             num_steps=5,
                             max_floor=25,
                             stack_size=2,
                             before_fc=[32],
                             actor_fc=[16],
                             critic_fc=[16],
                             lstm_size=32)
    ret = model.step(np.random.random(((1, 5,) + tuple(self.state_size))).astype(np.float32), np.zeros((1, lstm_size)), np.zeros((1, 5)))
    print(ret)
    print(ret[0].shape)
    print(type(ret[0]))
    print(ret[1].shape)
    print(type(ret[1]))
    print(ret[2].shape)
    print(type(ret[2]))
