# snake_game.py
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

Point = namedtuple("Point", "x, y")

BLOCK_SIZE = 20
DEBUG_WIDTH = 20 * BLOCK_SIZE

FAST_SPEED = 1000
SLOW_SPEED = 1000 / 144

W_WIDTH = 800
W_HEIGHT = 600

MAX_STEPS = 10
VISION_DIRECTIONS = [
    (20, 0),
    (-20, 0),
    (0, 20),
    (0, -20),
    (20, 20),
    (20, -20),
    (-20, 20),
    (-20, -20),
]


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class SnakeGameAI:
    def __init__(self, w=W_WIDTH, h=W_HEIGHT, num_food=5):
        pygame.font.init()
        pygame.display.set_caption("Snake AI")

        self.w = w
        self.h = h
        self.num_food = num_food
        self.display = pygame.display.set_mode((self.w + DEBUG_WIDTH, self.h))
        self.clock = pygame.time.Clock()
        self.reset()
        self.debug_info = {}
        self._render = True
        self._slowed = False
        self.font = pygame.font.SysFont("Verdana", 16)
        self.last_direction = Direction.RIGHT

        self.food_images = [
            pygame.transform.scale(pygame.image.load(
                "assets/apple.png").convert_alpha(), (BLOCK_SIZE, BLOCK_SIZE)),
            pygame.transform.scale(pygame.image.load(
                "assets/cooked_beef.png").convert_alpha(), (BLOCK_SIZE, BLOCK_SIZE))
        ]

    def reset(self):
        self.last_direction = self.direction = Direction.RIGHT
        self.old_head = self.head = Point(self.w / 2, self.h / 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - BLOCK_SIZE * 2, self.head.y),
        ]
        self.steps_since_food = 0
        self.old_head = self.head
        self.score = 0
        self.last_score = 0
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

    def _shutdown(self):
        pygame.quit()
        quit()

    def play_step(self, action):
        self.frame_iteration += 1
        self.last_score = self.score

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                pressed = pygame.key.get_pressed()
                if pressed[pygame.K_ESCAPE]:
                    self._shutdown()
                if pressed[pygame.K_INSERT]:
                    self._render = not self._render
                if pressed[pygame.K_UP]:
                    self._slowed = not self._slowed
            elif event.type == pygame.QUIT:
                self._shutdown()

        # Move snake
        self._move(action)
        self.snake.insert(0, self.head)
        self.steps_since_food += 1

        changed_direction = (self.direction != self.last_direction)

        # Check state
        game_over = False
        if self.is_collision() or self.frame_iteration > len(self.snake) * 500:
            game_over = True
            # Returns: done, score, food_eaten, changed_direction
            return game_over, self.score, False, changed_direction

        food_eaten = False
        for i, food in enumerate(self.foods):
            if self.head == food:
                self.foods.pop(i)
                self._place_food()

                self.score += 1

                food_eaten = True

                self.steps_since_food = 0
                break

        if not food_eaten:
            self.snake.pop()

        if self._render:
            self._update_ui()
            self.clock.tick(SLOW_SPEED if self._slowed else FAST_SPEED)

        return game_over, self.score, food_eaten, changed_direction

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        # Hits itself
        if pt in self.snake[1:]:
            return True

        # world boundary
        if (
            pt.x > self.w - BLOCK_SIZE
            or pt.x < 0
            or pt.y > self.h - BLOCK_SIZE
            or pt.y < 0
        ):
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
                    pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE,
                                BLOCK_SIZE, BLOCK_SIZE),
                )

        snake_parts = enumerate(self.snake)
        snake_len = len(self.snake)
        snake_head = self.snake[0]

        for i, pt in snake_parts:
            # Gradient blue color for snake body
            v = 100 if i == 0 else 0
            a = (i + 1) / snake_len
            clr = (v, v, 255 - (200 * a))

            pygame.draw.circle(self.display, clr, (pt.x + BLOCK_SIZE // 2, pt.y + BLOCK_SIZE // 2), BLOCK_SIZE * 0.4)

        for dx, dy in VISION_DIRECTIONS:
            pygame.draw.line(
                self.display,
                (0, 1, 0),
                (snake_head.x + 10, snake_head.y + 10),
                (snake_head.x + dx * MAX_STEPS, snake_head.y + dy * MAX_STEPS),
                1,
            )

        # Draw all food items
        counter = 0
        for food in self.foods:
            self.display.blit(self.food_images[counter & 1], pygame.Rect(
                food.x, food.y, BLOCK_SIZE, BLOCK_SIZE))
            counter += 1

        self.draw_debug_panel()

        pygame.display.flip()

    def draw_debug_panel(self):
        x_offset = self.w + 10
        y = 10

        self.draw_text("=== DEBUG PANEL ===", x_offset, y)
        y += 25

        if not self.debug_info:
            return

        self.draw_text(
            f"Epsilon: {self.debug_info['epsilon']:.2f}", x_offset, y)
        y += 18

        self.draw_text(
            f"Score/High: {self.debug_info['score']}/{self.debug_info['high_score']}", x_offset, y)
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
                text = ["Straight", "Right", "Left"][i]
                color = (
                    (0, 255, 0) if self.debug_info["action"] == i else (
                        200, 200, 200)
                )
                self.draw_text(f"{text}: {q[i]:.2f}", x_offset, y, color)
                y += 18

        self.draw_text(
            f"Chosen Action: {self.debug_info['action']}", x_offset, y)
        y += 18

        loss = self.debug_info.get("loss", None)
        if loss is not None:
            self.draw_text(f"Loss: {loss:.4f}", x_offset, y)
            y += 18

        groups = [
            ("Vision Rays", 24),    # 8 directions * 3 values
            ("Direction", 2),       # dir x, dir y
            ("Head Pos", 2),        # tail relative to head
            ("Body Size", 1),       
            ("Tail Pos", 2),        # dir_x, dir_y
            ("Food Vector", 2),     # food relative to head
            #("Head Delta", 6),
            #("Tail Delta", 6)
        ]

        self.draw_text("State Heatmap:", x_offset, y)
        y += 18

        state = self.debug_info["state"]

        idx = 0
        cols = 3
        cell = 18
        spacing = 2

        for name, length in groups:
            self.draw_text(name, x_offset, y)
            y += 22

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

            if y + box_h + 4 < self.h:
                y += box_h + 4
                print("abc")
            else: x_offset += box_w + 4; y = start_y

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN,
                      Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]
        elif np.array_equal(action, [0, 0, 1]):
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir
        self.old_head = self.head

        step = BLOCK_SIZE

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

    def draw_text(self, text, x, y, color=(255, 255, 255)):
        img = self.font.render(text, True, color)
        self.display.blit(img, (x, y))

    def draw_heatmap_cell(self, x, y, value, size=18):
        v = np.tanh(value)

        if v >= 0:
            color = (0, int(255 * v), 0)
        else:
            color = (int(255 * -v), 0, 0)

        pygame.draw.rect(self.display, color, pygame.Rect(x, y, size, size))

        pygame.draw.rect(self.display, (40, 40, 40),
                         pygame.Rect(x, y, size, size), 1)
