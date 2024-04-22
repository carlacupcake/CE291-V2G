
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Reshape
import random
import numpy as np


class DQNAgent:
    def __init__(self, state_size, action_size, model=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Experience replay buffer
        self.batch_size=24 #?
        self.gamma = .3  # discount rate tweak
        self.epsilon = .2  # exploration rate Tweak, no decay in this version
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995 # tweak
        self.learning_rate = 0.1

        if model is None:
            self.model = self._build_model()
        else:
            self.model = model
    
    def _build_model(self):
        model = tf.keras.Sequential([
            # Input layer:
            layers.Dense(50, activation='relu', input_shape=(self.state_size,)),
            # Hidden layers:
            layers.Dense(24, activation='relu'),
            layers.Dense(24, activation='relu'),  # You can add more layers or change the number of neurons
            # Output layer for action predictions:
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate))

        return model


    def act(self, current_state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        FFNN_input = np.expand_dims(current_state, axis=0)
        act_values = self.model.predict(FFNN_input, verbose=0)
        return np.argmax(act_values[0])  # returns action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def learn(self, current_state, action, reward, next_state, done):
        
        next_FFNN_input = np.expand_dims(next_state, axis=0)

        FFNN_input = np.expand_dims(current_state, axis=0)

        # Generate predictions for the next state to calculate the future Q-values
        if not done:
            next_qs = self.model.predict(next_FFNN_input, verbose=0)
            target = reward + self.gamma * np.amax(next_qs)  # Take the max Q-value among the next state predictions
        else:
            target = reward  # No future rewards if the episode is done

        # Generate predictions for the current state to update the targets
        target_f = self.model.predict(FFNN_input, verbose=0)
   
        # Update the Q-value for the action taken
        target_f[0][action] = target
        self.model.fit(FFNN_input, target_f, epochs=1, verbose=0)

