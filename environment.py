import math
import random

# Constant Values
BALL_X = 0
BALL_Y = 1
VEL_X = 2
VEL_Y = 3
PADDLE_Y = 4

PADDLE_HEIGHT = 0.2

STAY = 0
UP = 1
DOWN = 2


class Environment:

    def __init__(self):
        self.reward = 0
        self.time_step = 0
        self.state = [0.5, 0.5, 0.03, 0.01, 0.5 - PADDLE_HEIGHT / 2]

    def reset(self):
        self.reward = 0
        self.time_step = 0
        self.state = [0.5, 0.5, 0.03, 0.01, 0.5 - PADDLE_HEIGHT / 2]

    def make_time_step(self, action):
        """
        Make a Time Step by incrementing the time
            - Updating paddle position based on action (nothing, +0.04, -0.04)
            - Increment ballx by velocity x and ball y by velocity y
            - Bounce the ball if necessary
        :param action - a
        :return reward value of the time step
        """
        self.time_step = self.time_step + 1
        self.adjust_paddle(action)
        return self.adjust_ball()

    def adjust_paddle(self, action):
        """
        Adjust the paddle based on NOTHING, UP, DOWN
            NOTHING -> paddle maintains state
            UP -> paddley (top) is incremented by 0.04 (going down the array rep)
            DOWN -> paddle y(top) is decremented by 0.04 ( going up the array rep)
        :param action: int denoting the action to take
        """
        if action == STAY:
            return
        elif action == UP:  # Moving "DOWN" in the array rep
            new_val = self.state[PADDLE_Y] + 0.04
            if new_val + PADDLE_HEIGHT > 1:
                new_val = 1 - PADDLE_HEIGHT
            self.state[PADDLE_Y] = new_val
        else:
            new_val = self.state[PADDLE_Y] - 0.04
            if new_val < 0:
                new_val = 0
            self.state[PADDLE_Y] = new_val

    def adjust_ball(self):
        """
        Adjust the ball based on the current velocity
        :return -1 if the ball moved past the paddle, 1 if it bounces, 0 if neither
        """
        newx = self.state[BALL_X] + self.state[VEL_X]
        newy = self.state[BALL_Y] + self.state[VEL_Y]
        self.state[BALL_X] = newx
        self.state[BALL_Y] = newy
        # Off the top of the screen
        if newy < 0:
            self.state[BALL_Y] = -newy
            self.state[VEL_Y] = -self.state[VEL_Y]
        # Off the Bottom of the Screen
        if newy > 1:
            self.state[BALL_Y] = 2 - newy
            self.state[VEL_Y] = -self.state[VEL_Y]
        # Off Left Edge of Screen
        if newx < 0:
            self.state[BALL_X] = -newx
            self.state[VEL_X] = -self.state[VEL_X]
        # Bounced on the paddle or lost
        if self.state[BALL_X] >= 1:
            # Bounce
            if self.state[PADDLE_Y] <= self.state[BALL_Y] <= (self.state[PADDLE_Y] + PADDLE_HEIGHT):
                rand_x = random.uniform(-0.015, 0.015)
                rand_y = random.uniform(-0.03, 0.03)
                self.state[BALL_X] = 2 - self.state[BALL_X]
                self.state[VEL_X] = -self.state[VEL_X] + rand_x
                self.state[VEL_Y] = -self.state[VEL_Y] + rand_y

                # Check velocity
                if math.fabs(self.state[VEL_X]) < 0.03:
                    self.state[VEL_X] = 0.03 if self.state[VEL_X] > 0 else -0.03
                return 1
            # Lost - set state to 420 to indicate game over
            else:
                self.state[BALL_X] = 420
                return -1
        # Velocity Check
        if math.fabs(self.state[VEL_X]) < 0.03:
            self.state[VEL_X] = 0.03 if self.state[VEL_X] > 0 else -0.03
        return 0

    @staticmethod
    def simulate_transition(curr_state, action):
        """
        Simulate an action given the current state. Return the resulting state of that action
        :param curr_state: list denoting the state values of the initial state we want to simulate from
        :param action: int denoting the action to be taken
        :return: list denoting the resulting state of that action w.r.t the current state
        :return: int denoting the reward value of the resultant state
        """
        x, y, vx, vy, py = curr_state
        new_state = [x, y, vx, vy, py]
        # Adjust Paddle
        if action == UP:
            new_val = py + 0.04
            if new_val + PADDLE_HEIGHT > 1:
                new_val = 1 - PADDLE_HEIGHT
            new_state[PADDLE_Y] = new_val
        if action == DOWN:
            new_val = py - 0.04
            if new_val < 0:
                new_val = 0
            new_state[PADDLE_Y] = new_val
        # Adjust Ball
        newx = x + vx
        newy = y + vy
        new_state[BALL_X] = newx
        new_state[BALL_Y] = newy
        # Off the top of the screen
        if newy < 0:
            new_state[BALL_Y] = -newy
            new_state[VEL_Y] = -vy
        # Off the Bottom of the Screen
        if newy > 1:
            new_state[BALL_Y] = 2 - newy
            new_state[VEL_Y] = -vy
        # Off Left Edge of Screen
        if newx < 0:
            new_state[BALL_X] = -newx
            new_state[VEL_X] = -vx
        # Bounced on the paddle or lost
        if new_state[BALL_X] >= 1:
            # Bounce
            if new_state[PADDLE_Y] <= new_state[BALL_Y] <= new_state[PADDLE_Y] + PADDLE_HEIGHT:
                rand_x = random.uniform(-0.015, 0.015)
                rand_y = random.uniform(-0.03, 0.03)
                new_state[BALL_X] = 2 - new_state[BALL_X]
                new_state[VEL_X] = -new_state[VEL_X] + rand_x
                new_state[VEL_Y] = new_state[VEL_Y] + rand_y

                # Check velocity
                if math.fabs(new_state[VEL_X]) < 0.03:
                    new_state[VEL_X] = 0.03 if new_state[VEL_X] > 0 else -0.03
                return new_state, 1
            # Lost - set state to 420 to indicate game over
            else:
                return None, -1
        # Velocity Check
        if math.fabs(new_state[VEL_X]) < 0.03:
            new_state[VEL_X] = 0.03 if new_state[VEL_X] > 0 else -0.03
        return new_state, 0










