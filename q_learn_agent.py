import numpy as np
import math
import random
from environment import Environment

# Constants
BALL_X = 0
BALL_Y = 1
VEL_X = 2
VEL_Y = 3
PADDLE_Y = 4
PADDLE_HEIGHT = 0.2

STAY = 0
UP = 1
DOWN = 2


class QAgent:

    def __init__(self):
        self.q_table = np.zeros(shape=(12, 12, 2, 3, 12, 3))
        self.freq_table = np.zeros(shape=(12, 12, 2, 3, 12, 3))
        self.curr_state = None  # current state
        self.dead_reward = -1    # reward value / Q_value for the dead state
        self.num_bounces = 0    # reward for a current state
        self.gamma = 0.8
        self.alpha = 1
        self.epsilon = 0.2

    def test_agent(self, num_games):
        """
        Play the agent over a certain amount of test games to evaluate its learned policy
        :param num_games: int denoting number of games to be tested against
        :return: avg_score int denotign the average amount of times it is able to rebound the ball over numgames
        """
        test_environ = Environment()
        avg_score = 0
        for episode in range(0, num_games):
            cont_state = test_environ.state
            self.curr_state = self.get_discrete_state(cont_state)
            game_over = False
            while not game_over:
                # Simulate a time step
                x, y, vx, vy, py = self.curr_state
                action = np.argmax(self.q_table[x, y, vx, vy, py])
                next_state, reward = test_environ.simulate_transition(cont_state, action)
                # Action made a successful rebound
                if reward == 1:
                    avg_score = avg_score + 1
                # Game Over
                if next_state is None:
                    game_over = True
                # Keep going
                else:
                    cont_state = next_state
                    self.curr_state = self.get_discrete_state(cont_state)
        return avg_score / num_games

    def train_agent(self, num_games):
        """
        Train the Q-Learning Agent Over a Number of Games
        :param num_games: int denoting the number of games to train the agent on
        :return:
        """
        train_environ = Environment()
        data_points = int(num_games / 50)
        training_curve = np.zeros(shape=(data_points,))

        for episode in range(0, num_games):
            print("Starting Episode: " + str(episode))
            # Get the initial state
            cont_state = train_environ.state
            self.curr_state = self.get_discrete_state(cont_state)
            # Loop through the game
            game_over = False
            score = 0
            while not game_over:
                action, next_state, ns_reward = self.get_explore_action(cont_state, train_environ)
                if ns_reward == 1:
                    self.num_bounces = self.num_bounces + 1
                    score = score + 1
                # Check if the next_state ends in a terminal
                if next_state is None:
                    self.update_freq_table(action)
                    self.dead_update(action)
                    game_over = True
                # Game is live make proper updates
                else:
                    cont_state = next_state
                    self.update_freq_table(action)
                    self.update_max_action_val(next_state, action, ns_reward)
            # Update Data points if applicable
            if episode % 50 == 0:
                training_curve[int(episode / 50)] = self.num_bounces / episode if episode != 0 else 0
                print("Average : " + str(training_curve[int(episode/50)]))
            print("Episode : " + str(episode) + " Score : " + str(score) + " " + str(self.epsilon))

    def get_explore_action(self, cont_state, train_environ):
        """
        Get the next action from curr state and return the resulting state from that action using an exploration
        function
        :param cont_state: list denoting the continuous representation of a state
        :param train_environ: Environment instance
        :return: action int denoting the action chosen
        :return: next_state a continuous list representation of the state resultant from the action chosen
        :return: ns_reward int denoting the reward resultant from the transition
        """
        x, y, vx, vy, py = self.curr_state
        # Generate a random number for exploration
        p = random.random()
        # Choose randomly
        if p < self.epsilon:
            action = random.randint(0, 2)
        # Choose the maximum Q value of actions from the given state
        else:
            action = np.argmax(self.q_table[x, y, vx, vy, py])

        next_state, ns_reward = train_environ.simulate_transition(cont_state, action)
        return action, next_state, ns_reward

    def update_max_action_val(self, next_state, action, ns_reward):
        """
        Make a Q update using the value of the max action
            Q[s,a] <- Q[s,a] + alpha * (ns_reward + gamma * max(Q[s', a'] - Q[s, a]))
        Updates the current state as well
        :param next_state: continuous list representation of the next state resultant from an action
        :param action: action from curr state to next state
        :param ns_reward: reward of the new state
        """
        discrete_ns = self.get_discrete_state(next_state)
        x, y, vx, vy, py = self.curr_state
        dx, dy, dvx, dvy, dpy = discrete_ns

        # Compute the max action from the next state
        max_action_val = None
        for i in range(0, 3):
            curr_val = self.q_table[dx, dy, dvx, dvy, dpy, i]
            if max_action_val is None or curr_val > max_action_val:
                max_action_val = curr_val
        max_action_val = self.gamma * max_action_val - self.q_table[x, y, vx, vy, py, action]

        # Compute the total value
        learning_rate = self.alpha / (self.alpha + self.freq_table[x, y, vx, vy, py, action])
        lhs = learning_rate * (ns_reward + max_action_val)
        self.q_table[x, y, vx, vy, py, action] = self.q_table[x, y, vx, vy, py, action] + lhs
        self.curr_state = discrete_ns

    def update_freq_table(self, action):
        """
        Increment the frequency table based off of the current state action pair
        :param action: int denoting the action to be paired with the current state
        """
        x, y, vx, vy, py = self.curr_state
        self.freq_table[x, y, vx, vy, py, action] = self.freq_table[x, y, vx, vy, py, action] + 1

    def dead_update(self, action):
        """
        Update the current state for when it chooses an action leading to a game over
        :param action: int denoting the action to be made from the current state
        """
        x, y, vx, vy, py = self.curr_state
        dead_max = -self.q_table[x, y, vx, vy, py, action]
        learning_rate = self.alpha / (self.alpha + self.freq_table[x, y, vx, vy, py, action])
        lhs = learning_rate * (self.dead_reward + dead_max)
        self.q_table[x, y, vx, vy, py, action] = self.q_table[x, y, vx, vy, py, action] + lhs

    @staticmethod
    def get_discrete_state(state):
        """
        Return a discrete state version of the continuous state [x, y, vx, vy, py]
        x -> [0,1] to [0, 11]
        y -> [0,1] to [0, 11]
        vx, vy -> +1 or -1
        py -> floor(12 * paddle_y / 1-paddle_height))
        :param state - list of the continuous state values
        :return: the discrete state [x , y, vx, vy, py]
        """
        # Check for losing terminal state
        if state[BALL_X] == 420:
            return None
        # Otherwise Discretize as Defined
        dx = int(11 * state[BALL_X])
        dy = int(11 * state[BALL_Y])
        dvx = 1 if state[VEL_X] > 0 else -1
        if math.fabs(state[VEL_Y]) < 0.015:
            dvy = 0
        else:
            dvy = 1 if state[VEL_Y] > 0 else -1
        if state[PADDLE_Y] == (1 - PADDLE_HEIGHT):
            dpy = 11
        else:
            dpy = math.floor(12 * state[PADDLE_Y] / (1 - PADDLE_HEIGHT))
        return [dx, dy, dvx, dvy, dpy]
