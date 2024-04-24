
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Reshape
import random
import numpy as np


class DQNAgent:
    def __init__(self, state_size, action_size, sequence_length, model=None):
        self.state_size = state_size
        self.action_size = action_size
        self.sequence_length= sequence_length
        self.memory = deque(maxlen=2000)  # Experience replay buffer
        self.gamma = 0.01  # discount rate tweak
        self.epsilon = 1  # exploration rate Tweak, no decay in this version
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.95 # tweak
        self.learning_rate = 0.1
        
        if model is None:
            self.model = self._build_model()
        else:
            self.model = model
    """
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        #LSTM for temporal dependicies of data
        model = tf.keras.Sequential()
        model.add(layers.LSTM(50, input_shape=(self.sequence_length, self.state_size), return_sequences=True))
        model.add(layers.LSTM(50, return_sequences=False))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate))
        
        return model
        """


    def _build_model(self):
        model = tf.keras.Sequential([
            layers.SimpleRNN(50, input_shape=(self.sequence_length, self.state_size), return_sequences=True),
            layers.SimpleRNN(50, return_sequences=False),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='softmax')  # Using softmax for a probability distribution
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam')  # You might still use mse and adjust loss calculation manually
        return model


    def act(self, state_history):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        lstm_input = np.expand_dims(state_history, axis=0)
        act_values = self.model.predict(lstm_input, verbose=0)

        return np.argmax(act_values[0])  # returns action


    
    def learn(self, state_history, action, reward, future_state, done):
        lstm_input = np.expand_dims(state_history, axis=0)
        next_lstm_input = np.expand_dims(np.append(state_history[1:], [future_state], axis=0), axis=0)

        current_predictions = self.model.predict(lstm_input, verbose=0)
        next_predictions = self.model.predict(next_lstm_input, verbose=0)

        max_future_q = np.max(next_predictions[0]) if not done else 0
        target_q = reward + self.gamma * max_future_q

        target_f = current_predictions.copy()
        target_f[0, action] = target_q

        # Calculate the entropy of the predicted probabilities
        probabilities = current_predictions[0]
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-5))  # Small constant for numerical stability
        entropy_coefficient = 0.01  # Tweak this coefficient based on how much exploration you want

        # Use a custom training loop to apply the entropy-augmented loss
        with tf.GradientTape() as tape:
            logits = self.model(lstm_input, training=True)
            loss_value = tf.keras.losses.mean_squared_error(target_f, logits) - entropy_coefficient * entropy

        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

#PPO
#Actor Critic