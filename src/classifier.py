from src.agents.a3c import Memory, ActorCriticModel

class Classifier():
    def __init__(self,
                 num_steps=1000,
                 num_actions=3,
                 state_size=[1280],
                 stack_size=4,
                 learning_rate=0.00042,
                 batch_size=1000,
                 gamma=0.99,
                 update_freq=4,
                 actor_fc=None,
                 critic_fc=None,
                 summary_writer=None,
                 log_period=10,
                 checkpoint_period=10,
                 save_path="/tmp/a3c",
                 load_path=None,
                 data_path=None):

        self.save_path = save_path
        self.data_path = data_path
        self.summary_writer = summary_writer
        self.log_period = log_period
        self.checkpoint_period = checkpoint_period
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.num_episodes = num_episodes
        self.num_actions = num_actions
        self.state_size = state_size
        self.stack_size = stack_size
        self.batch_size=batch_size
        self.actor_fc = actor_fc
        self.critic_fc = critic_fc

        self.gamma = gamma

        self.opt = tf.keras.optimizers.Adam(learning_rate)

        self.model = ActorCriticModel(num_actions=self.num_actions,
                                      state_size=self.state_size,
                                      stack_size=self.stack_size,
                                      actor_fc=self.actor_fc,
                                      critic_fc=self.critic_fc)
        if load_path != None:
            self.model.load_weights(load_path)
            print("Loaded model from {}".format(load_path))

    
    def train():
        # Load human input from pickle file
        data_file = open(self.data_path, 'rb')
        memory_list = pickle.load(data_file)
        data_file.close()
        print("Loaded files")

        counts = [frame for memory in memory_list for frame in memory.actions]
        counts = [(len(counts) - c) / len(counts) for c in list(Counter(counts).values())]
        print("Counts: {}".format(counts))

        def gen():
            while True:
                for memory in memory_list:
                    for index, (action, state) in enumerate(zip(memory.actions, memory.states)):
                        stacked_state = [np.zeros_like(state) if index - i < 0 else memory.states[index - i].numpy()
                                      for i in reversed(range(self.stack_size))]
                        stacked_state = np.concatenate(stacked_state)
                        yield action, stacked_state

        print("Starting steps...")
        generator = gen()
        for train_step in range(self.num_steps):
            actions, states  = next(generator)
            weights = [counts[action] for action in actions]
            with tf.GradientTape() as tape:
                total_loss = self.compute_loss(actions,
                                               states,
                                               weights,
                                               self.gamma)
        
            # Calculate and apply policy gradients
            total_grads = tape.gradient(total_loss, self.model.actor_model.trainable_weights)
            self.opt.apply_gradients(zip(total_grads, self.model.actor_model.trainable_weights))
            print("Step: {} | Loss: {}".format(train_step, total_loss))

        critic_batch_size = 100
        critic_steps = 1000
        self.initialize_critic_model(critic_batch_size, critic_steps)

        _save_path = os.path.join(self.save_path, "human_trained_model.h5")
        os.makedirs(os.path.dirname(_save_path), exist_ok=True)
        self.model.save_weights(_save_path)
        print("Checkpoint saved to {}".format(_save_path))

    def compute_loss(self,
                     actions,
                     states,
                     weights,
                     gamma):

        # Get logits and values
        logits, _ = self.model(states)

        weighted_sparse_crossentropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        policy_loss = weighted_sparse_crossentropy(np.array(actions)[:, None], logits, sample_weight=weights)
        return policy_loss

    def initialize_critic_model(self, batch_size, critic_steps):
        random_states = np.random.random((batch_size,) + tuple(self.state_size[:-1] +
                                                            [self.state_size[-1] * self.stack_size]))
        zero_rewards = np.zeros(batch_size)
        for critic_step in range(critic_steps):
            with tf.GradientTape() as tape:
                values = self.model.critic_model(random_states)
                value_loss = keras.losses.mean_squared_error(zero_rewards[:, None], values)
            value_grads = tape.gradient(value_loss, self.model.critic_model.trainable_weights)

            self.opt.apply_gradients(zip(value_grads, self.model.critic_model.trainable_weights))

