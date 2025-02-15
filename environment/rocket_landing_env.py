import gym
from gym import spaces
import numpy as np
import time
import random
import pygame
import pygame.freetype


class RocketLandingEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(RocketLandingEnv, self).__init__()
        self.render_mode = render_mode

        # Rocket parameters
        self.screen_width = 800
        self.screen_height = 600
        self.rocket_width = 20
        self.rocket_height = 90
        self.gravity = 0.22
        self.damping = 0.99
        self.thrust = 0.5
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.ORANGE = (255, 165, 0)
        self.YELLOW = (255, 255, 0)
        self.GREEN = (0, 255, 0)

        # Define the action and observation space
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.clock = pygame.time.Clock()

        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Rocket Landing with DDQN")


        if self.render_mode == 'human':
            pygame.init()
            pygame.font.init()
            self.font = pygame.font.SysFont('Arial', 20)  # Choose a font and size
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Rocket Landing with DQN")
            self.clock = pygame.time.Clock()
        else:
            self.screen = None
            self.clock = None
            self.font = None


        self.rocket_surface = pygame.Surface((self.rocket_width, self.rocket_height), pygame.SRCALPHA)
        self.draw_rocket_shape()

        self.reset()


    def draw_rocket_shape(self):
        self.rocket_surface.fill((0, 0, 0, 0))  # Transparent background

        rocket_points = [
            (self.rocket_width / 2, 0),  # Tip of the rocket
            (self.rocket_width, self.rocket_height * 0.6),
            (self.rocket_width * 0.75, self.rocket_height),
            (self.rocket_width * 0.25, self.rocket_height),
            (0, self.rocket_height * 0.6)
        ]

        pygame.draw.polygon(self.rocket_surface, self.BLACK, rocket_points)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.rocket_pos = np.array([self.screen_width // 2, 50], dtype=float)
        self.rocket_velocity = np.array([0, 1], dtype=float)
        self.rocket_angle = 0
        self.landed = False
        self.crashed = False
        self.thrusting = False

        return self._get_observation()

    def _get_observation(self):
        return np.array([
            self.rocket_pos[0],
            self.rocket_pos[1],
            self.rocket_velocity[0],
            self.rocket_velocity[1],
            self.rocket_angle
        ], dtype=np.float32)

    def step(self, action):
        self.thrusting = False

        if action == 1:  # thrust
            self.rocket_velocity[1] -= self.thrust * np.cos(np.radians(self.rocket_angle))
            self.rocket_velocity[0] += self.thrust * np.sin(np.radians(self.rocket_angle))
            self.thrusting = True
        elif action == 2:  # Rotate counterclockwise
            self.rocket_angle += 2
        elif action == 3:  # Rotate clockwise
            self.rocket_angle -= 2

        # Apply gravity and damping
        self.rocket_velocity[1] += self.gravity
        self.rocket_velocity[0] *= self.damping
        self.rocket_pos += self.rocket_velocity

        reward = 0

        angle_penalty = -abs(self.rocket_angle) * 0.1
        reward += angle_penalty

        horizontal_velocity_penalty = -abs(self.rocket_velocity[0]) * 0.1
        reward += horizontal_velocity_penalty

        vertical_velocity_penalty = -max(0, self.rocket_velocity[1]) * 0.1
        reward += vertical_velocity_penalty

        if self.rocket_pos[0] < 0 or self.rocket_pos[0] > self.screen_width - self.rocket_width:
            reward -= 1000
            self.crashed = True
            done = True
            return self._get_observation(), reward, done, {}

        if self.rocket_pos[1] <= 0:
            self.rocket_pos[1] = 0
            reward -= 1000
            self.crashed = True
            done = True
            return self._get_observation(), reward, done, {}

        done = False
        if self.rocket_pos[1] >= self.screen_height - self.rocket_height:
            self.rocket_pos[1] = self.screen_height - self.rocket_height
            if abs(self.rocket_velocity[1]) <= 5.0 and abs(self.rocket_angle) <= 7:
                self.landed = True
                reward += 1000
            else:
                self.crashed = True
                crash_velocity = abs(self.rocket_velocity[1])
                penalty_scale = 2
                penalty = crash_velocity * penalty_scale
                reward -= penalty
            done = True

        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        if self.render_mode != 'human':
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        if self.screen is None:
            pygame.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Rocket Landing with DQN")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 20)

        self.draw_sky_gradient()

        ground_height = 30
        pygame.draw.rect(self.screen, (139, 69, 19),
                         (0, self.screen_height - ground_height, self.screen_width, ground_height))  # Brown ground

        border_thickness = 5
        pygame.draw.rect(self.screen, self.BLACK, (0, 0, self.screen_width, self.screen_height), border_thickness)

        if self.crashed:
            self.draw_crash(self.rocket_pos)
        elif self.landed:
            self.draw_rotated_rocket(self.rocket_surface, self.rocket_pos, self.rocket_angle)
            self.draw_success(self.rocket_pos)
        else:
            self.draw_rotated_rocket(self.rocket_surface, self.rocket_pos, self.rocket_angle)
            if self.thrusting:
                self.draw_ignition(self.rocket_pos)

        self.display_info()

        pygame.display.flip()
        self.clock.tick(30)  # 30 frames per second

    def display_info(self):
        velocity_text = f"Velocity: ({self.rocket_velocity[0]:.2f}, {self.rocket_velocity[1]:.2f})"
        angle_text = f"Angle: {self.rocket_angle:.2f}°"

        velocity_surface = self.font.render(velocity_text, True, self.BLACK)
        angle_surface = self.font.render(angle_text, True, self.BLACK)

        velocity_rect = velocity_surface.get_rect()
        angle_rect = angle_surface.get_rect()

        padding = 10
        total_height = velocity_rect.height + angle_rect.height + padding
        x_position = self.screen_width - velocity_rect.width - padding
        y_position = self.screen_height - total_height - padding

        self.screen.blit(velocity_surface, (x_position, y_position))
        self.screen.blit(angle_surface, (x_position, y_position + velocity_rect.height + padding // 2))

    def draw_sky_gradient(self):
        top_color = (135, 206, 250)
        bottom_color = (255, 255, 255)
        for y in range(self.screen_height):
            color_ratio = y / self.screen_height
            r = int(top_color[0] * (1 - color_ratio) + bottom_color[0] * color_ratio)
            g = int(top_color[1] * (1 - color_ratio) + bottom_color[1] * color_ratio)
            b = int(top_color[2] * (1 - color_ratio) + bottom_color[2] * color_ratio)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.screen_width, y))

    def draw_rotated_rocket(self, surface, pos, angle):
        rotated_surface = pygame.transform.rotate(surface, -angle)
        rotated_rect = rotated_surface.get_rect(
            center=(pos[0] + self.rocket_width // 2, pos[1] + self.rocket_height // 2))
        self.screen.blit(rotated_surface, rotated_rect.topleft)

    def draw_ignition(self, pos):
        flame_pos = (pos[0] + self.rocket_width // 2 - 4, pos[1] + self.rocket_height)
        flame_width = random.randint(8, 12)
        flame_height = random.randint(12, 20)
        pygame.draw.ellipse(self.screen, self.ORANGE, (*flame_pos, flame_width, flame_height))
        pygame.draw.ellipse(self.screen, self.RED, (flame_pos[0] + 2, flame_pos[1] + 8, 4, 8))

    def draw_crash(self, pos):
        for i in range(5, 50, 5):
            pygame.draw.circle(self.screen, self.RED,
                               (int(pos[0] + self.rocket_width // 2), int(pos[1] + self.rocket_height // 2)), i, 1)
            pygame.draw.circle(self.screen, self.ORANGE,
                               (int(pos[0] + self.rocket_width // 2), int(pos[1] + self.rocket_height // 2)), i // 2, 1)
            pygame.draw.circle(self.screen, self.YELLOW,
                               (int(pos[0] + self.rocket_width // 2), int(pos[1] + self.rocket_height // 2)), i // 4, 1)
        font = pygame.font.Font(None, 36)
        text = font.render("Rocket Crashed!", True, self.RED)
        self.screen.blit(text, (pos[0] - 50, pos[1] - 50))

    def draw_success(self, pos):
        font = pygame.font.Font(None, 36)
        text = font.render("Successful Landing!", True, self.GREEN)
        self.screen.blit(text, (pos[0] - 50, pos[1] - 50))

    def close(self):
        if self.render_mode == 'human' and self.screen is not None:
            pygame.quit()

