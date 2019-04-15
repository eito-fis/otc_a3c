
import numpy as np
import tensorflow as tf

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
        self.floors = []

    def store(self, state, action, reward, floor):
        # Save state - action - reward pairs
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.floors.append(floor)

    def clear(self):
        # Wipe data
        self.states = []
        self.actions = []
        self.rewards = []
        self.obs=[]
        self.floors = []

class ActorCriticModel(tf.keras.Model):
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

        # Build general input and any additional input specific to models
        self.model_input = tf.keras.layers.Input(shape=tuple(state_size))
        self.critic_input = tf.keras.layers.Input(shape=(1,))

        # Build the fully connected layers for the actor and critic models
        actor_x = self.model_input
        for neurons in actor_fc:
            actor_x = tf.keras.layers.Dense(neurons, activation="relu")(actor_x)
        critic_x = tf.keras.layers.concatenate([self.model_input, self.critic_input])
        for neurons in critic_fc:
            critic_x = tf.keras.layers.Dense(neurons, activation="relu")(critic_x)

        # Build the output layers for the actor and critic models
        self.actor_logits = tf.keras.layers.Dense(num_actions, name='policy_logits')(actor_x)
        self.value = tf.keras.layers.Dense(1, name='value')(critic_x)

        # Build the final actor and critic models
        self.actor_model = tf.keras.models.Model(inputs=[self.model_input], outputs=[self.actor_logits])
        self.critic_model = tf.keras.models.Model(inputs=[self.model_input, self.critic_input], outputs=[self.value])

        # Run each model with random inputs to force Keras to build the static graphs
        self.actor_model(np.random.random((1,) + tuple(state_size)).astype(np.float32))
        self.critic_model([np.random.random((1,) + tuple(state_size)).astype(np.float32),
                           np.random.random((1, 1)).astype(np.float32)])
        self.predict([np.random.random((1,) + tuple(state_size)).astype(np.float32),
                      np.random.random((1, 1)).astype(np.float32)])

    def call(self, inputs):
        # Split input list into observation data and floor data
        obs, floor = inputs

        # Call each model on the input, and return each output
        actor_logits = self.actor_model(obs)
        value = self.critic_model([obs, floor])

        return actor_logits, value

    def get_action_value(self, obs, floor):
        '''
        Samples action and returns scalar values instead of tensors
        '''
        # Call each model on the input
        logits, value  = self.predict([obs, floor])

        # Sample from the actor model distribution, then convert to scalar values
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=-1).numpy()
        action = action[0]
        value = value[0][0]

        return action, value
