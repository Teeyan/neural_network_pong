import pygame, sys
from pygame.locals import *
from behavioral_cloning import BCAgent
from environment import Environment


pygame.init()
FPS = 100

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 76, 76)
BASICFONTSIZE = 20
BASICFONT = pygame.font.Font('freesansbold.ttf', BASICFONTSIZE)

# Overhead
BALL_X = 0
BALL_Y = 1
VEL_X = 2
VEL_Y = 3
PADDLE_Y = 4

PADDLE_HEIGHT = 0.2

WIDTH = 600
HEIGHT = 600
LINETHICKNESS = 10

# canvas declaration
window = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)


# Draw The Canvas
def draw_arena():
    window.fill(BLACK)
    pygame.draw.line(window, WHITE, (0, 0), (0, HEIGHT), LINETHICKNESS * 4)


# Draw a paddle
def draw_paddle():
    paddle = pygame.Rect(WIDTH - LINETHICKNESS, BALL_STATE[PADDLE_Y] * HEIGHT, LINETHICKNESS, PADDLE_HEIGHT * HEIGHT)
    pygame.draw.rect(window, WHITE, paddle)


# Draw the ball
def draw_ball():
    ball = pygame.Rect(BALL_STATE[BALL_X] * WIDTH, BALL_STATE[BALL_Y] * HEIGHT, LINETHICKNESS, LINETHICKNESS)
    pygame.draw.rect(window, WHITE, ball)


# Display and Update Score
def display_score(score):
    result_surf = BASICFONT.render('Score = %s' % score, True, WHITE)
    result_rect = result_surf.get_rect()
    result_rect.topleft = (WIDTH - 100, 10)
    window.blit(result_surf, result_rect)


# Reflect a Game Over Screen Until The user hits restart
def draw_game_end():
    restart_button = pygame.Rect(WIDTH*0.25, HEIGHT*0.4, WIDTH * 0.5, HEIGHT * 0.2)
    restart_font = BASICFONT.render('GAME OVER', True, BLACK)
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


def main():
    # Overhead
    pygame.init()
    FPS_CLOCK = pygame.time.Clock()
    pygame.display.set_caption("GIGADRILLBRKRZ")
    oracle = Environment()

    # Game Objects
    global BALL_STATE
    BALL_STATE = [0.5, 0.5, 0.03, 0.01, 0.5 - PADDLE_HEIGHT / 2]
    SCORE = 0

    agent = BCAgent()
    agent.load_parameters()

    # Game Loop
    while True:
        # Event Loop
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        # Get the Agent's action at this time state and update our Ball State
        action = agent.make_move(BALL_STATE)
        BALL_STATE, reward = oracle.simulate_transition(BALL_STATE, action)

        # Check for game end
        if BALL_STATE is None:
            draw_game_end()
            # Reset the Ball state
            BALL_STATE = [0.5, 0.5, 0.03, 0.01, 0.5 - PADDLE_HEIGHT / 2]
            SCORE = 0

        # Update Score
        if reward == 1:
            SCORE = SCORE + 1

        # Draw the game board
        draw_arena()
        # Draw the paddle
        draw_paddle()
        # Draw the ball
        draw_ball()

        display_score(SCORE)

        pygame.display.update()
        FPS_CLOCK.tick(FPS)

if __name__ == "__main__":
    main()
