import pygame, sys
from pygame.locals import *
from behavioral_cloning import BCAgent
import math
import random


pygame.init()
FPS = 100

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 76, 76)
BASICFONTSIZE = 20
BASICFONT = pygame.font.Font('freesansbold.ttf', BASICFONTSIZE)
STAY = 0
UP = 1
DOWN = 2

# Overhead
BALL_X = 0
BALL_Y = 1
VEL_X = 2
VEL_Y = 3
PADDLE_Y = 4

PADDLE_HEIGHT = 0.2
PADDLE_D = 0

WIDTH = 1000
HEIGHT = 600
PLAYER_PADDLE = 0
PLAYER_WALL = WIDTH/2
CPU_PADDLE = WIDTH
CPU_WALL = WIDTH/2
LINETHICKNESS = 10

# 0 is player Turn | 1 is cpu turn
TURN = 1

# canvas declaration
window = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)


def draw_arena():
    window.fill(BLACK)
    pygame.draw.line(window, WHITE, (WIDTH/2, 0), (WIDTH/2, HEIGHT), LINETHICKNESS)


def draw_paddle(paddle):
    pygame.draw.rect(window, WHITE, paddle)


def draw_ball(ball):
    pygame.draw.rect(window, WHITE, ball)


def update_player_paddle():
    PLAYER_BALL_STATE[PADDLE_Y] = PLAYER_BALL_STATE[PADDLE_Y] + PADDLE_D
    if PLAYER_BALL_STATE[PADDLE_Y] + PADDLE_HEIGHT > 1:
        PLAYER_BALL_STATE[PADDLE_Y] = 1 - PADDLE_HEIGHT
    if PLAYER_BALL_STATE[PADDLE_Y] < 0:
        PLAYER_BALL_STATE[PADDLE_Y] = 0
    return pygame.Rect(0, PLAYER_BALL_STATE[PADDLE_Y] * HEIGHT, LINETHICKNESS, PADDLE_HEIGHT * HEIGHT)


def update_cpu_paddle():
    return pygame.Rect(WIDTH - LINETHICKNESS, BALL_STATE[PADDLE_Y] * HEIGHT, LINETHICKNESS, PADDLE_HEIGHT * HEIGHT)


def update_ball_cpu_side():
    return pygame.Rect((WIDTH / 2) + (BALL_STATE[BALL_X] * (WIDTH/2)), BALL_STATE[BALL_Y] * HEIGHT, LINETHICKNESS, LINETHICKNESS)


def update_ball_player_side():
    return pygame.Rect((WIDTH / 2) * PLAYER_BALL_STATE[BALL_X], PLAYER_BALL_STATE[BALL_Y] * HEIGHT, LINETHICKNESS, LINETHICKNESS)


# Display and Update Score
def display_score(p_score, cpu_score):
    result_surf = BASICFONT.render('%s' % p_score, True, WHITE)
    result_rect = result_surf.get_rect()
    result_rect.topleft = (WIDTH/2 - 20, 10)
    cpu_surf = BASICFONT.render('%s' % cpu_score, True, WHITE)
    cpu_rect = cpu_surf.get_rect()
    cpu_rect.topleft = (WIDTH/2 + 10, 10)
    window.blit(result_surf, result_rect)
    window.blit(cpu_surf, cpu_rect)


def draw_game_end():
    restart_button = pygame.Rect(WIDTH*0.25, HEIGHT*0.4, WIDTH * 0.5, HEIGHT * 0.2)
    restart_font = BASICFONT.render('NEXT GAME', True, BLACK)
    pygame.draw.rect(window, RED, restart_button)
    window.blit(restart_font, (WIDTH * 0.4, HEIGHT * 0.48))
    pygame.display.update()
    # Loop until an event has occurred
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if restart_button.collidepoint(event.pos):
                    return


# Key Handlers
def keydown(event):
    global PADDLE_D
    if event.key == K_UP:
        PADDLE_D = -0.04
    elif event.key == K_DOWN:
        PADDLE_D = 0.04
    else:
        PADDLE_D = 0


def keyup(event):
    global PADDLE_D
    if event.key in (K_UP, K_DOWN):
        PADDLE_D = 0


def simulate_player(curr_state):
    x, y, vx, vy, py = curr_state
    new_state = [x, y, vx, vy, py]
    # Adjust Ball
    newx = x + vx
    newy = y + vy
    new_state[BALL_X] = newx
    new_state[BALL_Y] = newy
    # Off the top of the screen
    if newy < 0:
        new_state[BALL_Y] = -newy
        new_state[VEL_Y] = -vy
    # Off the bottom of the screen
    if newy > 1:
        new_state[BALL_Y] = 2 - newy
        new_state[VEL_Y] = -vy
    # BOUNCED OFF RIGHT EDGE OF SCREEN
    if newx > 1:
        new_state[BALL_X] = 1 - newx
        if math.fabs(new_state[VEL_X]) < 0.03:
            new_state[VEL_X] = 0.03 if new_state[VEL_X] > 0 else -0.03
        return new_state, 420
    # Bounced on the paddle or lost
    if new_state[BALL_X] <= 0:
        # Bounce
        if new_state[PADDLE_Y] <= new_state[BALL_Y] <= new_state[PADDLE_Y] + PADDLE_HEIGHT:
            rand_x = random.uniform(-0.015, 0.015)
            rand_y = random.uniform(-0.03, 0.03)
            new_state[BALL_X] = 0 - new_state[BALL_X]
            new_state[VEL_X] = -new_state[VEL_X] + rand_x
            new_state[VEL_Y] = new_state[VEL_Y] + rand_y

            # Check velocity
            if math.fabs(new_state[VEL_X]) < 0.03:
                new_state[VEL_X] = 0.03 if new_state[VEL_X] > 0 else -0.03
            return new_state, 1
        # Lost
        else:
            return None, -1
    # Velocity Check
    if math.fabs(new_state[VEL_X]) < 0.03:
        new_state[VEL_X] = 0.03 if new_state[VEL_X] > 0 else -0.03
    return new_state, 0


def simulate_cpu(curr_state, action):
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

    # BOUNCED OFF LEFT EDGE OF SCREEN
    if newx < 0:
        new_state[BALL_X] = 1 - (-newx)
        if math.fabs(new_state[VEL_X]) < 0.03:
            new_state[VEL_X] = 0.03 if new_state[VEL_X] > 0 else -0.03
        return new_state, 420

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


# Main Game Loop
def main():
    # Overhead
    pygame.init()
    FPS_CLOCK = pygame.time.Clock()
    pygame.display.set_caption("GIGADRILLBRKRZ")

    # Game Objects
    global PLAYER_BALL_STATE, BALL_STATE, TURN
    BALL_STATE = [0.5, 0.5, 0.03, 0.01, 0.5 - PADDLE_HEIGHT / 2]
    PLAYER_BALL_STATE = [0, 0, 0, 0, 0.5 - PADDLE_HEIGHT / 2]
    PLAYER_SCORE = 0
    CPU_SCORE = 0
    player_paddle = pygame.Rect(0, HEIGHT * PLAYER_BALL_STATE[PADDLE_Y], LINETHICKNESS, PADDLE_HEIGHT * HEIGHT)
    cpu_paddle = pygame.Rect(WIDTH - LINETHICKNESS, HEIGHT * BALL_STATE[PADDLE_Y], LINETHICKNESS, PADDLE_HEIGHT * HEIGHT)
    ball = pygame.Rect(WIDTH * 0.75, HEIGHT * 0.5, LINETHICKNESS, LINETHICKNESS)

    # Initialize Agent
    agent = BCAgent()
    agent.load_parameters()

    # Game Loop
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                keydown(event)
            if event.type == KEYUP:
                keyup(event)

        # Update Player Paddle - Asynchronous to turns
        player_paddle = update_player_paddle()

        # CPU has the ball
        if TURN == 1:
            # Get Agent Action
            action = agent.make_move(BALL_STATE)
            BALL_STATE, reward = simulate_cpu(BALL_STATE, action)
            # On Human side of the court, handle appropriately
            if reward == 420:
                PLAYER_BALL_STATE[0] = BALL_STATE[BALL_X]
                PLAYER_BALL_STATE[1] = BALL_STATE[BALL_Y]
                PLAYER_BALL_STATE[2] = BALL_STATE[VEL_X]
                PLAYER_BALL_STATE[3] = BALL_STATE[VEL_Y]
                TURN = 0
                ball = update_ball_player_side()
            # Check for game over
            elif reward == -1:
                PLAYER_SCORE = PLAYER_SCORE + 1
                draw_game_end()
                BALL_STATE = [0.5, 0.5, 0.03, 0.01, 0.5 - PADDLE_HEIGHT / 2]
                PLAYER_BALL_STATE = [0, 0, 0, 0, 0.5 - PADDLE_HEIGHT / 2]
                player_paddle = pygame.Rect(0, HEIGHT * PLAYER_BALL_STATE[PADDLE_Y], LINETHICKNESS, PADDLE_HEIGHT * HEIGHT)
                cpu_paddle = pygame.Rect(WIDTH - LINETHICKNESS, HEIGHT * BALL_STATE[PADDLE_Y], LINETHICKNESS,
                                         PADDLE_HEIGHT * HEIGHT)
                ball = pygame.Rect(WIDTH * 0.75, HEIGHT * 0.5, LINETHICKNESS, LINETHICKNESS)
                continue
            # CPU SIDE
            else:
                ball = update_ball_cpu_side()
                cpu_paddle = update_cpu_paddle()

        # Player has the ball
        else:
            # Make Time Step
            PLAYER_BALL_STATE, reward = simulate_player(PLAYER_BALL_STATE)
            # On CPU side of the court, handle appropriately
            if reward == 420:
                BALL_STATE[BALL_X] = PLAYER_BALL_STATE[BALL_X]
                BALL_STATE[BALL_Y] = PLAYER_BALL_STATE[BALL_Y]
                BALL_STATE[VEL_X] = PLAYER_BALL_STATE[VEL_X]
                BALL_STATE[VEL_Y] = PLAYER_BALL_STATE[VEL_Y]
                TURN = 1
                ball = update_ball_cpu_side()

            # Check for game over
            elif reward == -1:
                CPU_SCORE = CPU_SCORE + 1
                draw_game_end()
                BALL_STATE = [0.5, 0.5, 0.03, 0.01, 0.5 - PADDLE_HEIGHT / 2]
                PLAYER_BALL_STATE = [0, 0, 0, 0, 0.5 - PADDLE_HEIGHT / 2]
                player_paddle = pygame.Rect(0, HEIGHT * PLAYER_BALL_STATE[PADDLE_Y], LINETHICKNESS, PADDLE_HEIGHT * HEIGHT)
                cpu_paddle = pygame.Rect(WIDTH - LINETHICKNESS, HEIGHT * BALL_STATE[PADDLE_Y], LINETHICKNESS,
                                         PADDLE_HEIGHT * HEIGHT)
                ball = pygame.Rect(WIDTH * 0.75, HEIGHT * 0.5, LINETHICKNESS, LINETHICKNESS)
                TURN = 1
                continue

            # Human side
            else:
                ball = update_ball_player_side()

        # Draw Graphics
        draw_arena()
        draw_paddle(player_paddle)
        draw_paddle(cpu_paddle)
        draw_ball(ball)

        display_score(PLAYER_SCORE, CPU_SCORE)

        pygame.display.update()
        FPS_CLOCK.tick(FPS)

if __name__ == "__main__":
    main()
