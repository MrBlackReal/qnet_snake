import torch
import random
import numpy as np
from collections import deque
from snake_game import Direction, Point
from net import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

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


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(35, 256, 4)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.last_prediction = None
        self.last_action = None

    def look_in_direction(self, game, direction):
        head = game.snake[0]
        x, y = head.x, head.y
        dx, dy = direction

        food_seen = 0
        body_seen = 0
        distance = 0

        while True:
            x += dx
            y += dy
            distance += 1
            pt = Point(x, y)

            if game.is_collision(pt):
                return food_seen, body_seen, 1 / distance

            if pt in game.foods:
                food_seen = 1

            if pt in game.snake[1:]:
                body_seen = 1

    def get_state(self, game):
        head = game.snake[0]
        tail = game.snake[-1]

        # Direction vector
        dir_x = 0
        dir_y = 0

        if game.direction == Direction.RIGHT:
            dir_x = 1
        if game.direction == Direction.LEFT:
            dir_x = -1
        if game.direction == Direction.DOWN:
            dir_y = 1
        if game.direction == Direction.UP:
            dir_y = -1

        # Closest food
        food = min(game.foods, key=lambda f: abs(f.x - head.x) + abs(f.y - head.y))

        state = []

        # Vision rays
        for d in VISION_DIRECTIONS:
            state.extend(self.look_in_direction(game, d))
        
        # Head position
        state.append(head.x / game.w)
        state.append(head.y / game.h)

        # Body size
        #state.append(len(game.snake) / (game.w * game.h))
        state.append((tail.x - head.x))
        state.append((tail.y - head.y))

        # Tail position
        state.append(head.x / game.w)
        state.append(head.y / game.h)

        # Direction
        state.append(dir_x)
        state.append(dir_y)

        # Food vector
        state.append((food.x - head.x) / game.w)
        state.append((food.y - head.y) / game.h)

        # Boost Awareness
        state.append(1 if game.did_boost else 0)

        return np.array(state, dtype=np.float32)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = max(5, 150 - self.n_games)
        final_move = [0, 0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

            self.last_prediction = prediction.detach().numpy()
            self.last_action = move

        final_move[move] = 1
        return final_move
