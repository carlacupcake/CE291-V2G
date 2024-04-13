
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import random
import numpy as np


class DQNAgent_nr:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Experience replay buffer
        self.gamma = 0.95  # discount rate tweak
        self.epsilon = 1.0  # exploration rate Tweak
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995 # tweak
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        # TWEAK, because of time dependcy, would a LSTM work better?
        #use some sort of convolutional NN
        model = tf.keras.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate))

        return model


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state_batch_dim = np.expand_dims(state, axis=0)  # Add a batch dimension

        act_values = self.model.predict(state_batch_dim, verbose=0)
        return np.argmax(act_values[0])  # returns action

    def learn(self, current_state, action, reward, future_state, done):
        state = np.expand_dims(current_state, axis=0)  # Adding batch dimension
        next_state = np.expand_dims(future_state, axis=0)  # Adding batch dimension
        target = reward
        if not done:
            target = (reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0]))
        target_f = self.model.predict(state, verbose=0)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
#PPO
#Actor Critic