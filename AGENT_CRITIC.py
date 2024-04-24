
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Reshape
from tensorflow.keras import layers, models, optimizers
import random
import numpy as np


class AGENT_CRITIC:
    def __init__(self, state_size, action_size, sequence_length, actormodel=None, criticmodel=None):
        self.state_size = state_size
        self.action_size = action_size
        self.sequence_length= sequence_length
        self.gamma = 0  # discount rate tweak
        self.learning_rate = 0.01
        self.epsilon=.1

        if actormodel is None:
            self.actor = self._build_actor()
        else:
            self.actor = actormodel

        if criticmodel is None:
            self.critic = self._build_critic()
        else:
            self.critic = criticmodel


    def _build_actor(self):
        # Actor model that predicts action probabilities
        model = models.Sequential([
            layers.LSTM(50, input_shape=(self.sequence_length, self.state_size), return_sequences=True),
            layers.LSTM(50),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='softmax')  # Output probabilities for each action
        ])
        #model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate))

        return model

    def _build_critic(self):
        # Critic model that predicts the value of state-action pairs
        model = models.Sequential([
            layers.LSTM(50, input_shape=(self.sequence_length, self.state_size), return_sequences=True),
            layers.LSTM(50),
            layers.Dense(24, activation='relu'),
            layers.Dense(1)  # Output a single value for state-value
        ])
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            # Otherwise, select an action based on the policy's output probabilities
            state = state[np.newaxis, :]  # Reshape state for prediction
            probabilities = self.actor.predict(state, verbose=0)[0]
            return np.random.choice(self.action_size, p=probabilities)


    def learn(self, state_history, action, reward, next_state, done):
        # Ensure the input shape is correct for a single instance
        next_state_history = np.append(state_history[1:], [next_state], axis=0)  # make sequence for future state

        lstm_input = np.expand_dims(state_history, axis=0)
        next_lstm_input = np.expand_dims(next_state_history, axis=0)

        # Predict the value of the current and next state
        critic_value = self.critic.predict(lstm_input, verbose=0)[0]
        critic_value_next = self.critic.predict(next_lstm_input, verbose=0)[0]

        # If not done, calculate discounted target reward
        target = reward
        if not done:
            target += self.gamma * critic_value_next  # Updating this to use the first element directly

        # Convert target to the correct shape for training
        target = np.array([[target]])  # Shape (1, 1) to match output shape of critic

        # Train the critic
        self.critic.fit(lstm_input, target, epochs=1, verbose=0)

        # Train the actor
        with tf.GradientTape() as tape:
            # Get the prediction for the current state
            actions = self.actor(lstm_input, training=True)

            # Calculate log probabilities
            action_masks = tf.one_hot([action], self.action_size)  # Use a list to correctly shape the tensor
            log_probs = tf.reduce_sum(tf.one_hot([action], self.action_size) * tf.math.log(actions + 1e-8), axis=1)
            #log_probs = tf.reduce_sum(action_masks * tf.math.log(actions), axis=1)

            # Compute advantages
            advantages = target - critic_value

            # Compute loss
            loss = -tf.reduce_mean(log_probs * advantages)

        # Compute and apply gradients
        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))


""" 
    def learn(self, state_history, action, reward, next_state, done):
        # Ensure the input shape is correct for a single instance
        next_state_history = np.append(state_history[1:], [next_state], axis=0)  # make sequnce for future state

        lstm_input = np.expand_dims(state_history, axis=0)
        next_lstm_input = np.expand_dims(next_state_history, axis=0)

        # Predict the value of the current and next state
        critic_value = self.critic.predict(lstm_input, verbose=0)
        critic_value_next = self.critic.predict(next_lstm_input, verbose=0)

        # If not done, calculate discounted target reward
        target = reward
        if not done:
            target = reward + self.gamma * critic_value_next[0]

        # Convert target to the correct shape for training
        target = np.array([[target]])  # Shape (1, 1) to match output shape of critic

        # Train the critic
        self.critic.fit(lstm_input, target, epochs=1, verbose=0)

        # Train the actor
        with tf.GradientTape() as tape:
            tape.watch(self.actor.trainable_variables)
            actions = self.actor(lstm_input, training=True)
            action_prob = actions[0, action]
            log_prob = tf.math.log(action_prob)
            loss = -log_prob * (target - critic_value)

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))


        """
