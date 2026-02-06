import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

Point = namedtuple("Point", "x, y")

DEBUG_WIDTH = 300
BLOCK_SIZE = 20
SPEED = 60


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class SnakeGameAI:
    def __init__(self, w=800, h=600, num_food=10):
        pygame.font.init()
        pygame.display.set_caption("Snake AI Training")

        self.w = w
        self.h = h
        self.num_food = num_food
        self.display = pygame.display.set_mode((self.w + DEBUG_WIDTH, self.h))
        self.clock = pygame.time.Clock()
        self.reset()
        self.debug_info = {}
        self.render = True
        self.font = pygame.font.SysFont("Verdana", 16)

        self.fruit_image = pygame.image.load("assets/apple.png").convert_alpha()
        self.fruit_image = pygame.transform.scale(self.fruit_image, (BLOCK_SIZE, BLOCK_SIZE))

        self.food_images = [pygame.image.load("assets/apple.png").convert_alpha(), pygame.image.load("assets/cooked_beef.png").convert_alpha()]

        img = self.food_images.pop()
        while img is not None:
            self.food_images.append()
            img = pygame.transform.scale(self.food_images.pop(), (BLOCK_SIZE, BLOCK_SIZE))

        for img in self.food_images:
            img = self.food_images.pop()
            img = pygame.transform.scale(img, (BLOCK_SIZE, BLOCK_SIZE))


    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
        ]
        self.did_boost = False
        self.steps_since_food = 0
        self.old_head = self.head
        self.score = 0
        self.foods = []

        for _ in range(self.num_food):
            self._place_food()

        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        food = Point(x, y)
        if food in self.snake or food in self.foods:
            self._place_food()
        else:
            self.foods.append(food)

    def play_step(self, action):
        self.frame_iteration += 1
        self.steps_since_food += 1

        # 1. Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Move
        self._move(action)
        self.snake.insert(0, self.head)

        # 3. Check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. Place new food or just move
        food_eaten = False
        for i, food in enumerate(self.foods):
            if self.head == food:
                self.score += 1
                reward = 10
                self.foods.pop(i)
                self._place_food()
                food_eaten = True
                break

        if not food_eaten:
            self.snake.pop()
        else:
            speed_bonus = max(0, 5 - self.steps_since_food) * 0.5
            reward += speed_bonus
            self.steps_since_food = 0

        closest_food = min(
            self.foods, key=lambda f: abs(f.x - self.head.x) + abs(f.y - self.head.y)
        )

        old_dist = abs(self.old_head.x - closest_food.x) + abs(
            self.old_head.y - closest_food.y
        )
        new_dist = abs(self.head.x - closest_food.x) + abs(self.head.y - closest_food.y)

        if new_dist < old_dist:
            reward += 0.2
        else:
            reward -= 0.2

        if self.did_boost and len(self.snake) > 3:
            self.snake.pop()
            reward += 0.5

        if self.did_boost and len(self.snake) <= 3:
            reward -= 1.2  # discourage suicidal boosting

        # 5. Update UI and clock
        if self.render:
            self._update_ui()
            self.clock.tick(SPEED)

        # 6. Return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Hits boundary
        if (
            pt.x > self.w - BLOCK_SIZE
            or pt.x < 0
            or pt.y > self.h - BLOCK_SIZE
            or pt.y < 0
        ):
            return True
        # Hits itself
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill((40, 40, 40))

        for y in range(self.h // BLOCK_SIZE):
            for x in range(self.w // BLOCK_SIZE):
                color = (35, 35, 35) if (x + y) & 1 == 0 else (30, 30, 30)
                pygame.draw.rect(
                    self.display,
                    color,
                    pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE),
                )

        snake_parts = enumerate(self.snake)
        snake_len = len(self.snake)
        snake_head = self.snake[0]

        for i, pt in snake_parts:
            # Gradient blue color for snake body
            clr = 100 if i == 0 else 0

            pygame.draw.rect(
                self.display,
                (clr, clr, 255 - (200 * (i + 1) // snake_len)),
                pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE),
            )

        for dx, dy in [
            (20, 0),
            (-20, 0),
            (0, 20),
            (0, -20),
            (20, 20),
            (20, -20),
            (-20, 20),
            (-20, -20),
        ]:
            pygame.draw.line(
                self.display,
                (0, 1, 0),
                (snake_head.x + 10, snake_head.y + 10),
                (snake_head.x + dx * 6, snake_head.y + dy * 6),
                1,
            )

        # Draw all food items
        for food in self.foods:
            pygame.draw.rect(
                self.display,
                (255, 0, 0),
                pygame.Rect(food.x, food.y, BLOCK_SIZE, BLOCK_SIZE),
            )

        self.draw_debug_panel()

        pygame.display.flip()

    def draw_debug_panel(self):
        x_offset = self.w + 10
        y = 10

        self.draw_text("=== DEBUG PANEL ===", x_offset, y)
        y += 25

        if not self.debug_info:
            return

        self.draw_text(f"Epsilon: {self.debug_info['epsilon']:.2f}", x_offset, y)
        y += 18

        reward = self.debug_info.get("reward", 0)
        color = (
            (0, 255, 0)
            if reward > 0
            else (255, 0, 0)
            if reward < 0
            else (200, 200, 200)
        )
        self.draw_text(
            f"Reward: {' ' if reward > 0 else ''}{reward:.2f}", x_offset, y, color
        )
        y += 18

        self.draw_text("Q-Values:", x_offset, y)
        y += 18

        q = self.debug_info["q_values"]

        if q is not None:
            for i, qv in enumerate(q):
                text = ["Straight", "Right", "Left", "Boost"][i]
                color = (
                    (0, 255, 0) if self.debug_info["action"] == i else (200, 200, 200)
                )
                self.draw_text(f"{text}: {q[i]:.2f}", x_offset, y, color)
                y += 18

        self.draw_text(f"Chosen Action: {self.debug_info['action']}", x_offset, y)
        y += 18

        groups = [
            ("Vision Rays", 24),
            ("Head Pos", 2),
            ("Body Size", 2),
            ("Tail Pos", 2),
            ("Direction", 2),
            ("Food Vec", 2),
            ("Boost", 1)
        ]

        self.draw_text("State Heatmap:", x_offset, y)
        y += 18

        state = self.debug_info["state"]

        idx = 0
        cols = 6
        cell = 18
        spacing = 2

        for name, length in groups:
            self.draw_text(name, x_offset, y)
            y += 20

            start_y = y

            for i in range(length):
                v = state[idx]
                cx = x_offset + (i % cols) * (cell + spacing)
                cy = y + (i // cols) * (cell + spacing)
                self.draw_heatmap_cell(cx, cy, v, cell)
                idx += 1

            rows = (length - 1) // cols + 1
            box_w = cols * (cell + spacing)
            box_h = rows * (cell + spacing)

            pygame.draw.rect(
                self.display,
                (80, 80, 80),
                (x_offset - 4, start_y - 4, box_w + 6, box_h + 6),
                1
            )

            y += box_h + 10


    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        boost = False

        if np.array_equal(action, [1, 0, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0, 0]):
            new_dir = clock_wise[(idx + 1) % 4]
        elif np.array_equal(action, [0, 0, 1, 0]):
            new_dir = clock_wise[(idx - 1) % 4]
        else:
            new_dir = clock_wise[idx]
            boost = True

        self.direction = new_dir
        self.did_boost = boost
        self.old_head = self.head

        step = BLOCK_SIZE * (2 if boost else 1)

        x, y = self.head.x, self.head.y

        if self.direction == Direction.RIGHT:
            x += step
        elif self.direction == Direction.LEFT:
            x -= step
        elif self.direction == Direction.DOWN:
            y += step
        elif self.direction == Direction.UP:
            y -= step

        self.head = Point(x, y)

        if self.old_head is None:
            self.old_head = self.head

    def draw_text(self, text, x, y, color=(255, 255, 255)):
        img = self.font.render(text, True, color)
        self.display.blit(img, (x, y))

    def draw_heatmap_cell(self, x, y, value, size=18):
        # Clamp value for visualization
        v = np.tanh(value)

        if v >= 0:
            color = (0, int(255 * v), 0)
        else:
            color = (int(255 * -v), 0, 0)

        pygame.draw.rect(self.display, color, pygame.Rect(x, y, size, size))

        pygame.draw.rect(self.display, (40, 40, 40), pygame.Rect(x, y, size, size), 1)
