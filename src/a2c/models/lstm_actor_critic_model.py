
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class Custom_LSTM(layers.Layer):
    def __init__(self, lstm_size, input_size, ortho_scale):
        super(Custom_LSTM, self).__init__()
        self.lstm_size = lstm_size
        self.input_size = input_size
        self.ortho_scale = ortho_scale

    def build(self, input_shape):
        self.weight_x = self.add_weight(name="input_weights",
                                        shape=(input_shape[0][-1], self.lstm_size * 4),
                                        initializer=tf.initializers.Orthogonal(gain=self.ortho_scale),
                                        trainable=True)
        self.weight_h = self.add_weight(name="hidden_weights",
                                        shape=(self.lstm_size, self.lstm_size * 4),
                                        initializer=tf.initializers.Orthogonal(gain=self.ortho_scale),
                                        trainable=True)
        self.bias = self.add_weight(name="lstm_bias",
                                    shape=(self.lstm_size * 4,),
                                    initializer=tf.initializers.Constant(0),
                                    trainable=True)

    def call(self, input_tensor, cell_state_hidden, mask_tensor):
        cell_state, hidden = tf.split(axis=1, num_or_size_splits=2, value=cell_state_hidden)
        for idx, (_input, mask) in enumerate(zip(input_tensor, mask_tensor)):
            # Reset cell state and hidden if our this timestep is a restart
            cell_state = cell_state * (1 - mask)
            hidden = hidden * (1 - mask)
            
            # Run LSTM cell!
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

class LSTMActorCriticModel(tf.keras.models.Model):
    '''
    LSTMActor Critic Model
    Arguments:
    num_actions: Number of logits the actor model will output
    state_size: List containing the expected size of the state
    max_floor: Maximium number of floors reachable
    stack_size: Numer of states we can expect in our stack
    actor_fc: Iterable containing the amount of neurons per layer for the actor model
    critic_fc: Iterable containing the amount of neurons per layer for the critic model
        ex: (1024, 512, 256) would make 3 fully connected layers, with 1024, 512 and 256
            layers respectively
    '''
    def __init__(self,
                 num_actions=None,
                 state_size=None,
                 num_steps=None,
                 num_envs=4,
                 max_floor=25,
                 stack_size=None,
                 before_fc=None,
                 actor_fc=None,
                 conv_size=None,
                 critic_fc=None,
                 lstm_size=None,
                 retro=True):
        super(LSTMActorCriticModel, self).__init__()

        self.num_actions = num_actions
        self.actor_fc = actor_fc
        self.critic_fc = critic_fc
        self.lstm_size = lstm_size
        self.num_steps = num_steps
        self.num_envs = num_envs

        # Multiply the final dimension of the state by stack_size to get correct input_size
        self.state_size = state_size[:-1] + [state_size[-1] * stack_size]

        # Build convolutional layer
        if conv_size is not None:
            if isinstance(conv_size, tuple):
                self.convs = Custom_Convs(conv_size)
            elif conv_size == "quake":
                self.convs = Quake_Block()
            else:
                raise ValueError("Invalid CNN Topology")
            self.flatten = layers.Flatten()
        else:
            self.convs = None
        
        # Build fully connected layers that go between convs and LSTM
        self.before_fc = [layers.Dense(neurons, activation="relu", name="before_dense_{}".format(i)) for i,(neurons) in enumerate(before_fc)]

        # Build LSTM
        self.lstm = Custom_LSTM(lstm_size, num_steps, 1.0)

        # Build the fully connected layers for the actor and critic models
        self.actor_fc = [layers.Dense(neurons, activation="relu", name="actor_dense_{}".format(i)) for i,(neurons) in enumerate(actor_fc)]
        self.critic_fc = [layers.Dense(neurons, activation="relu", name="critic_dense_{}".format(i)) for i,(neurons) in enumerate(critic_fc)]

        # Build the output layers for the actor and critic models
        self.actor_logits = layers.Dense(num_actions, name='policy_logits')
        self.value = layers.Dense(1, name='value')

        # Take a step with random input to build the model
                            
        self.step([(np.random.random(tuple(self.state_size)).astype(np.float32),
                  np.random.random((max_floor + 1,)).astype(np.float32)) for _ in range(self.num_envs)],
                  np.zeros((num_envs, lstm_size * 2)).astype(np.float32),
                  np.zeros((num_envs,)).astype(np.float32))

    def call(self, inputs):
        all_obs, cell_state_hidden, reset_mask = inputs
        image_obs, data_obs = all_obs

        if self.convs:
            conv_x = self.convs(image_obs)
            before_x = self.flatten(conv_x)
        else:
            before_x = obs
        for l in self.before_fc:
            before_x = l(before_x)

        lstm_x = self.batch_to_seq(before_x)
        reset_mask = self.batch_to_seq(reset_mask, flat=True)
        lstm_x, cell_state_hidden = self.lstm(lstm_x, cell_state_hidden, reset_mask)
        lstm_x = self.seq_to_batch(lstm_x)

        actor_x = lstm_x
        for l in self.actor_fc:
            actor_x = l(actor_x)
        logits = self.actor_logits(actor_x)

        critic_x = layers.concatenate([lstm_x, data_obs])
        for l in self.critic_fc:
            critic_x = l(critic_x)
        value = self.value(critic_x)

        return (logits, value, cell_state_hidden)

    def step(self, inputs, cell_state_hidden, reset_mask):

        # Make predictions on input
        inputs = self.process_inputs(inputs)
        logits, values, cell_state_hidden = self([inputs, cell_state_hidden, reset_mask])
        probs = tf.nn.softmax(logits)

        # Sample from probability distributions
        actions = tf.squeeze(tf.random.categorical(logits, 1), axis=-1).numpy()

        # Get action probabilities
        one_hot_actions = tf.one_hot(actions, self.num_actions)
        action_probs = tf.reduce_sum(probs * one_hot_actions, axis=-1).numpy()

        values = np.squeeze(values)

        return actions, values, action_probs, cell_state_hidden.numpy()

    def get_values(self, inputs, cell_state_hidden, reset_mask):
        inputs = self.process_inputs(inputs)
        _, values, _ = self([inputs, cell_state_hidden, reset_mask])
        values = tf.squeeze(values)

        return values.numpy()

    def process_inputs(self, inputs, multi_input=True):
        '''
        Convert n_envs x n_inputs list to n_inputs x n_envs list if we have
        multiple inputs, otherwise just convert to array
        '''
        if multi_input:
            inputs = [np.asarray(l) for l in zip(*inputs)]
        else:
            inputs = np.array([l for l in inputs])
        return inputs

    def batch_to_seq(self, tensor_batch, flat=False):
        '''
        Converts a flattened full batch of tensors into a sequence of tensors
        Used to convert our batch into a form our LSTM Cell can iterate over and understand
        '''
        shape = tensor_batch.get_shape().as_list()
        # If batch size is the number of environments, we are stepping and only have 1 step
        if shape[0] == self.num_envs:
            n_steps = 1
        else:
            n_steps = self.num_steps
        if flat:
            input_size = [1]
        else:
            input_size = shape[1:]
        tensor_batch = tf.reshape(tensor_batch, [-1, n_steps] + input_size)
        return [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=n_steps, value=tensor_batch)]

    def seq_to_batch(self, tensor_sequence, flat=False):
        '''
        Converts a sequence of tensors into a batch of tensors
        '''
        shape = tensor_sequence[0].get_shape().as_list()
        if not flat:
            assert len(shape) > 1
            return tf.reshape(tf.concat(axis=1, values=tensor_sequence), [-1, shape[-1]])
        else:
            return tf.reshape(tf.stack(values=tensor_sequence, axis=1), [-1])

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
        super(Quake_Block, self).__init__(name='')

        # Quake 3 Deepmind style convolutions
        # Like Nature CNN but with an additional 3x3 kernel and skip connections
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
