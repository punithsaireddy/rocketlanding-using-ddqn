import pygame
import sys
from environment.rocket_landing_env import RocketLandingEnv

def main():
    env = RocketLandingEnv()
    clock = pygame.time.Clock()

    obs = env.reset()
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                pygame.quit()
                sys.exit()

        keys = pygame.key.get_pressed()

        # Mapping keys to actions
        if keys[pygame.K_UP]:
            action = 1  # Apply thrust
        elif keys[pygame.K_LEFT]:
            action = 2  # Rotate left
        elif keys[pygame.K_RIGHT]:
            action = 3  # Rotate right
        else:
            action = 0  # Do nothing

        # Taking a step in the environment
        obs, reward, done, info = env.step(action)

        # Rendering the environment
        env.render()

        # Control the frame rate
        clock.tick(30)  # Limit to 30 FPS

    env.close()

if __name__ == "__main__":
    main()
