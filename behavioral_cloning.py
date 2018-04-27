import numpy as np
import scipy.special
import random
from environment import Environment

UP = 0
STAY = 1
DOWN = 2


class BCAgent:

    def __init__(self):
        self.num_bounces = 0
        self.data_mean = np.zeros(shape=(5,))
        self.data_std = np.zeros(shape=(5,))
        self.data = self.read_in_training_data("expert_policy.txt")
        self.weights = self.initialize_weights(4, 128)
        self.bias = self.initialize_bias(4, 128)
        self.learning_rate = 0.1
        self.bias_lr = 0.01

    def evaluate_on_train(self):
        """
        Evaluate the agent on the training data
        :return: float denoting training accuracy
        """
        correct = 0
        for i in range(0, len(self.data)):
            if self.get_action(self.data[i, :-1]) == self.data[i, -1]:
                correct = correct + 1
        return correct / len(self.data)

    def test_agent(self, num_games):
        """
        Test the agent over the course of num_games games
        :param num_games: int denoting the number of test games to run the agent through
        :return: avg_score: float denoting the average number of bounces in the test games
        """
        test_environ = Environment()
        avg_score = 0
        for game in range(0, num_games):
            curr_state = test_environ.state
            game_over = False
            while not game_over:
                # Simulate a time_step
                # Normalize the current state
                processed_state = np.zeros(shape=(5,))
                for i in range(0, len(curr_state)):
                    processed_state[i] = (curr_state[i] - self.data_mean[i]) / self.data_std[i]
                # Get the proper action given the normalized state
                action = self.get_action(processed_state)
                translated_action = None
                if action == UP:
                    translated_action = 2
                elif action == STAY:
                    translated_action = 0
                else:
                    translated_action = 1
                # Map to correct action in environment
                # print(translated_action, curr_state)
                next_state, reward = test_environ.simulate_transition(curr_state, translated_action)
                if reward == 1:
                    avg_score = avg_score + 1
                if next_state is None:
                    game_over = True
                else:
                    curr_state = next_state
        return avg_score / num_games

    def get_action(self, test_x):
        """
        Get the proper action (classification) given a game state (observation)
        :param test_x: numpy array of size 5 denoting the game state
        :return: int denoting the action to take
        """
        # Forward Propogation
        # Input -> Input Layer -> Z1 output of input layer
        z_first = self.affine_forward(test_x, self.weights[0], self.bias[0])
        # Z1 -> ReLU -> A1 input of first layer
        a_first = self.relu_forward(z_first)
        # A1 -> First Layer -> Z2 output of first layer
        z_second = self.affine_forward(a_first, self.weights[1], self.bias[1])
        # Z2 -> ReLu -> A2 input of second layer
        a_second = self.relu_forward(z_second)
        # A2 -> Second Layer -> Z3 output of second layer
        z_third = self.affine_forward(a_second, self.weights[2], self.bias[2])
        # Z3 -> ReLu -> A3 input of third layer
        a_third = self.relu_forward(z_third)
        # A3 ->  Output Layer -> final output
        outputs = self.affine_forward(a_third, self.weights[3], self.bias[3])
        return np.argmax(outputs)

    def make_move(self, x):
        """
        Given a list of floats denoting a state, get the proper action to make
        :param x: list of size 5 denoting the game state
        :return: int denoting the action to take
        """
        # Scale the input
        scaled = np.zeros(shape=(5,))
        for i in range(0, len(scaled)):
            scaled[i] = (x[i] - self.data_mean[i]) / self.data_std[i]
        # Get action
        action = self.get_action(scaled)
        if action == UP:
            return 2
        elif action == STAY:
            return 0
        else:
            return 1

    def train_agent(self, num_epochs, batch_size):
        """
        Train Agent through minibatch gradient descent
        :param num_epochs: int denoting number of epochs to train through
        :param batch_size: int denoting number of observations in a batch
        """
        for epoch in range(0, num_epochs):
            # Shuffle the Data
            self.data = np.random.permutation(self.data)
            # Iterate through this batch for training
            for batch in range(0, int(len(self.data)/batch_size)):
                start_ind = batch * batch_size
                batch_feat = self.data[start_ind:start_ind + batch_size, :-1]
                batch_y = self.data[start_ind:start_ind + batch_size, -1]
                loss, zero, one, two = self.train_network(batch_feat, batch_y)
                print("Epoch : " + str(epoch) + " Batch : " + str(batch) + " Loss : " + str(loss) + " Accuracy: " + \
                      str(zero) + " | " + str(one) + " | " + str(two))

        # Save the parameters
        self.save_parameters()

    def train_network(self, batch_feats, batch_y):
        """
        Train the network as a three layer network
        :param batch_feats: features of the batch dataset (numpy 2d n x 5)
        :param batch_y: response variable of the batch dataset (numpy 1d n x 1)
        :return: loss the difference between the network output and actual test labels
        """
        # Forward Propogation
        # Input -> Input Layer -> Z1 output of input layer
        z_first = self.affine_forward(batch_feats, self.weights[0], self.bias[0])
        # Z1 -> ReLU -> A1 input of first layer
        a_first = self.relu_forward(z_first)
        # A1 -> First Layer -> Z2 output of first layer
        z_second = self.affine_forward(a_first, self.weights[1], self.bias[1])
        # Z2 -> ReLu -> A2 input of second layer
        a_second = self.relu_forward(z_second)
        # A2 -> Second Layer -> Z3 output of second layer
        z_third = self.affine_forward(a_second, self.weights[2], self.bias[2])
        # Z3 -> ReLu -> A3 input of third layer
        a_third = self.relu_forward(z_third)
        # A3 -> Output Layer -> Output
        outputs = self.affine_forward(a_third, self.weights[3], self.bias[3])

        # Calculate the accuracy
        pred_zero = 0
        zero_n = 0
        pred_one = 0
        one_n = 0
        pred_two = 0
        two_n = 0
        view = np.argmax(outputs, axis=1)
        for i in range(0, len(batch_y)):
            if batch_y[i] == 0:
                zero_n = zero_n + 1
                if view[i] == 0:
                    pred_zero = pred_zero + 1
            elif batch_y[i] == 1:
                one_n = one_n + 1
                if view[i] == 1:
                    pred_one = pred_one + 1
            else:
                two_n = two_n + 1
                if view[i] == 2:
                    pred_two = pred_two + 1

        # Compute the Loss and the differential of the outputs
        loss, dlogits = self.cross_entropy(outputs, batch_y)

        # Backwards Propogation
        # Output -> Third Layer
        da_3, dw_4, db_4 = self.affine_backwards(dlogits, a_third, self.weights[3])
        dz_3 = self.relu_backward(da_3, z_third)
        # Output -> Second Layer plus ReLu
        da_2, dw_3, db_3, = self.affine_backwards(dz_3, a_second, self.weights[2])
        dz_2 = self.relu_backward(da_2, z_second)
        # Second Layer -> First Layer
        da_1, dw_2, db_2 = self.affine_backwards(dz_2, a_first, self.weights[1])
        dz_1 = self.relu_backward(da_1, z_first)
        # First layer -> Input
        dx, dw_1, db_1 = self.affine_backwards(dz_1, batch_feats, self.weights[0])

        # Update parameters with gradient descent
        self.weights[0] = self.weights[0] - (self.learning_rate * dw_1)
        self.weights[1] = self.weights[1] - (self.learning_rate * dw_2)
        self.weights[2] = self.weights[2] - (self.learning_rate * dw_3)
        self.weights[3] = self.weights[3] - (self.learning_rate * dw_4)
        self.bias[0] = self.bias[0] - (self.bias_lr * db_1)
        self.bias[1] = self.bias[1] - (self.bias_lr * db_2)
        self.bias[2] = self.bias[2] - (self.bias_lr * db_3)
        self.bias[3] = self.bias[3] - (self.bias_lr * db_4)
        return loss, (pred_zero / zero_n), (pred_one / one_n), (pred_two / two_n)

    def save_parameters(self):
        """
        Save the weights and biases in to a text file
        """
        np.savetxt("weights1.txt", self.weights[0])
        np.savetxt("weights2.txt", self.weights[1])
        np.savetxt("weights3.txt", self.weights[2])
        np.savetxt("weights4.txt", self.weights[3])
        np.savetxt("bias1.txt", self.bias[0])
        np.savetxt("bias2.txt", self.bias[1])
        np.savetxt("bias3.txt", self.bias[2])
        np.savetxt("bias4.txt", self.bias[3])

    def load_parameters(self):
        """
        Load in the weights and biases from a text file
        :return:
        """
        self.weights[0] = np.loadtxt("weights1.txt")
        self.weights[1] = np.loadtxt("weights2.txt")
        self.weights[2] = np.loadtxt("weights3.txt")
        self.weights[3] = np.loadtxt("weights4.txt")
        self.bias[0] = np.loadtxt("bias1.txt")
        self.bias[1] = np.loadtxt("bias2.txt")
        self.bias[2] = np.loadtxt("bias3.txt")
        self.bias[3] = np.loadtxt("bias4.txt")

    @staticmethod
    def affine_forward(data, weights, bias):
        """
        Compute an affine transformation on the data in forward propogation where d' is the number of layer units
        :param data: 2D numpy array of shap n x d
        :param weights: layer weight matrix np array of shape d x d'
        :param bias: bias array b -> np array of shape (d',)
        :return: Z - affine output of the transformation 2d numpy array of shape n x d'
        """
        z = np.dot(data, weights)
        return z + bias

    @staticmethod
    def affine_backwards(diff_z, data, weights):
        """
        Compute the gradients of the loss L with respect to the forward propogation inputs A, W, b
        :param diff_z: gradient dZ - 2d numpy array of shape n x d'
        :param data: the affine output of the affine forward operation
        :param weights: layer weight matrix np array of shape d x d'
        :return: dA - gradient dA w.r.t. the loss - 2d numpy array of shape n x d
        :return: dW - gradient dW w.r.t the loss - 2d numpy array of shape d x d'
        :return: db - gradient of the bias - numpy array of shape (d',)
        """
        # Calculate the gradient of the data
        dA = np.dot(diff_z, np.swapaxes(weights, 0, 1))
        # Calculate the gradient of the weights
        dW = np.dot(np.swapaxes(data, 0, 1), diff_z)
        # Calculate the gradient of the bias
        db = np.sum(diff_z, 0)

        return dA, dW, db

    @staticmethod
    def relu_forward(z):
        """
        Compute the elementwise ReLu of Z where a relu is simply
            xi = { xi for xi > 0 | 0 otherwise }
        :param z: batch z matrix, 2d numpy array of size n x d'
        :return: ReLU output, 2d array of size n x d'
        """
        relu_z = np.copy(z)
        relu_z[relu_z < 0] = 0
        return relu_z

    @staticmethod
    def relu_backward(diff_a, z_og):
        """
        Computes gradient of Z with respect to loss. Z and the a are the same shape
        :param diff_a: differential of the data (zeroed out)
        :param z_og: original z matrix data
        :return: gradient of z with respect to the loss L
        """
        diff_z = np.copy(diff_a)
        for i in range(0, len(z_og)):
            for j in range(0, len(z_og[0])):
                if z_og[i, j] < 0:
                    diff_z[i, j] = 0
        return diff_z

    @staticmethod
    def cross_entropy(f, y):
        """
        Computes the loss function L and the gradients of the loss w.r.t the scores F
        :param f: logits scores for the predictions np array of size (n, 3)
        :param y: target classes for the observations - np array of size (n,)
        :return: loss L and the gradient dlogits
        """
        n = len(y)
        # Compute the loss L and the gradient dlogits
        loss = 0
        dlogits = np.zeros(shape=f.shape)
        for i in range(0, n):
            # loss computation using logsumexp for stability
            loss = loss + (f[i, int(y[i])] - scipy.special.logsumexp(f[i,:]))
            # gradient dlogits computation
            for j in range(0, len(f[0])):
                inner = 1 if j == y[i] else 0
                inner = inner - (np.exp(f[i, j]) / np.sum(np.exp(f[i,:])))
                dlogits[i, j] = (-1 / n) * inner
        loss = loss * (-1 / n)
        return loss, dlogits

    @staticmethod
    def initialize_bias(num_layers, num_units):
        """
        Initialize bias values to all zeros
        :param num_layers: int denoting the number of weight matrices needed (one per layer)
        :param num_units : int denoting the number of features for a hidden layer
        :return: list of numpy arrays denoting the biases for each layer
        """
        # Initial bias array should be size num_units
        bias = []
        for i in range(0, num_layers - 1):
            bias.append(np.zeros(shape=(num_units,)))
        # Output bias should be size 3
        bias.append(np.zeros(shape=(3,)))
        return bias

    @staticmethod
    def initialize_weights(num_layers, num_units):
        """
        Initialize weight matrices to random values between 0 and 0.1
        :param num_layers: int denoting the number of weight matrices needed (one per layer)
        :param num_units: int denoting the number of features for a hidden layer
        :return: numpy list of numpy matrices denoting the weight matrices
        """
        # Initial weight matrix should be size 5 x 256
        weights = [np.random.uniform(low=-1, high=1, size=(5, num_units)) * 0.1]
        # Hidden layer weight matrices should be size 256 x 256
        for i in range(1, num_layers - 1):
            weights.append(np.random.uniform(low=-1, high=1, size=(num_units, num_units)) * 0.1)
        # Output layer weight matrix should be size 256 x 3
        weights.append(np.random.uniform(low=-1, high=1, size=(num_units, 3)) * 0.1)
        return weights

    def read_in_training_data(self, filename):
        """
        Read in the training data from the txt file
        :param filename: string denoting location of the text file
        :return: 2d numpy array of observations and features -> ball state and response variable
        """
        # Read in data into the numpy array
        data = np.loadtxt(fname=filename, delimiter=" ")
        # Normalize the data
        for column in range(0, 5):
            mean = np.mean(data[:, column])
            std = np.std(data[:, column])
            self.data_mean[column] = mean
            self.data_std[column] = std
            data[:, column] = (data[:, column] - mean) / std
        return data
