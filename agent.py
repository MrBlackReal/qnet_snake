# agent.py
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque
from snake_game import Direction, Point, VISION_DIRECTIONS, MAX_STEPS, BLOCK_SIZE, W_WIDTH, W_HEIGHT
from net import Linear_QNet, Conv_QNet
from utils import manhattan_distance, manhattan_distance_blocks

from config import MAX_MEMORY, BATCH_SIZE, LR

class QTrainer:
    def __init__(self, model, target_model, lr, gamma):
        self.model = model
        self.target_model = target_model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.last_loss = 0.0  # store last loss

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        pred = self.model(state)
        target = pred.clone().detach()

        for i in range(len(done)):
            q_new = reward[i]
            if not done[i]:
                # Use target network for stability
                q_new = reward[i] + self.gamma * \
                    torch.max(self.target_model(next_state[i]))
            target[i][torch.argmax(action[i]).item()] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        self.last_loss = loss.item()  # save for debug


class Agent:
    def __init__(self, epsilon=1, gamma=0.99):
        self.n_games = 0
        self.epsilon = epsilon  # Randomness
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.gamma = gamma
        self.hidden_size = 256

        self.memory = deque(maxlen=MAX_MEMORY)
        self.head_history = deque(maxlen=4)
        self.tail_history = deque(maxlen=4)

        self.model = Linear_QNet(33, self.hidden_size, self.hidden_size, 3)
        self.target_model = Linear_QNet(
            33, self.hidden_size, self.hidden_size, 3)

        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.trainer = QTrainer(
            self.model, self.target_model, lr=LR, gamma=self.gamma)

        self.last_prediction = None
        self.last_action = None
        self.time_facing_same_dir = 0
        self.target_food = None

    def update_target_network(self, tau=0.01):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(
                tau * param.data + (1.0 - tau) * target_param.data)

    def look_in_direction(self, game, direction, max_steps=MAX_STEPS):
        head = game.snake[0]
        x, y = head.x, head.y
        dx, dy = direction

        food_seen = 0
        body_seen = 0
        steps = 0

        while True:
            steps += 1
            x += dx
            y += dy
            pt = Point(x, y)

            if pt in game.snake[1:]:
                body_seen = 1
            elif pt in game.foods:
                food_seen = 1

            if game.is_collision(pt) or steps >= max_steps:
                dist_score = 1.0 / steps
                return food_seen, body_seen, dist_score

    def get_state(self, game):
        head = game.snake[0]
        tail = game.snake[-1]

        self.head_history.append(head)
        self.tail_history.append(tail)

        while len(self.head_history) < 4:
            self.head_history.append(head)
        while len(self.tail_history) < 4:
            self.tail_history.append(tail)

        # Direction vector
        dir_x = 0
        dir_y = 0

        if game.direction == Direction.RIGHT:
            dir_x = 1
        elif game.direction == Direction.LEFT:
            dir_x = -1
        elif game.direction == Direction.DOWN:
            dir_y = 1
        elif game.direction == Direction.UP:
            dir_y = -1

        # Closest food
        if game.last_score != game.score or game.score <= 0:
            self.target_food = min(
                game.foods, key=lambda f: manhattan_distance(head, f))

        state = []

        # 1. Vision rays (food_seen, body_seen, distance_score)
        for d in VISION_DIRECTIONS:
            state.extend(self.look_in_direction(game, d))

        # 4. Current direction
        state.append(dir_x)
        state.append(dir_y)

        # 2. Head position normalized
        state.append(head.x / game.w)
        state.append(head.y / game.h)

        # feel of body size
        state.append(len(game.snake) / 100)

        # 3. Tail vector relative to head
        state.append((tail.x - head.x) / game.w)
        state.append((tail.y - head.y) / game.h)

        # 5. Food vector relative to head
        state.append((self.target_food.x - head.x) / game.w)
        state.append((self.target_food.y - head.y) / game.h)

        return np.array(state, dtype=np.float32)

    def circular_motion_penalty(self):
        if len(self.head_history) < 4:
            return 0.0

        p0, p1, p2, p3 = self.head_history
        v1 = np.array([p1.x - p0.x, p1.y - p0.y], dtype=float)
        v2 = np.array([p2.x - p1.x, p2.y - p1.y], dtype=float)
        v3 = np.array([p3.x - p2.x, p3.y - p2.y], dtype=float)

        # protect against zero-length vectors
        limit = 1e-6
        if np.linalg.norm(v1) < limit or np.linalg.norm(v2) < limit or np.linalg.norm(v3) < limit:
            return 0.0

        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        v3 /= np.linalg.norm(v3)

        # compute turning magnitude (1 - dot) so straight -> 0, sharp turn -> larger
        turn1 = 1.0 - np.dot(v1, v2)
        turn2 = 1.0 - np.dot(v2, v3)

        # if it is turning repeatedly, penalize; otherwise small bonus for straightness
        if turn1 > 0.5 and turn2 > 0.5:
            return -0.08  # penalize repeated sharp turns

        # reward straight-ish motion
        straightness = (1.0 - turn1) * (1.0 - turn2)
        return 0.02 * straightness

    def compute_reward(self, game, done, food_eaten, changed_direction):
        """
        Stable, normalized reward. Inputs:
        - game: SnakeGameAI
        - done: boolean (game over)
        - food_eaten: boolean (food was consumed this step)
        - changed_direction: boolean (we turned this step)
        """
        # Immediate terminal
        if done:
            return -20.0

        reward = 0.0

        head = game.snake[0]

        # --- 1) Food reward (single strong, but not massive) ---
        if food_eaten:
            reward += 8.0   # kept sizable but not huge

        # --- 2) Small step survival bonus (encourage staying alive) ---
        reward += 0.01

        # --- 3) Smoothness: small encouragement for straight motion, small penalty for jitter ---
        if changed_direction:
            reward -= 0.02   # small, not crippling
        else:
            reward += 0.01

        # --- 4) Safety: immediate free-neighbor count (0..4) -> encourage safe moves ---
        free_neighbors = 0
        for dx, dy in [(BLOCK_SIZE, 0), (-BLOCK_SIZE, 0), (0, BLOCK_SIZE), (0, -BLOCK_SIZE)]:
            pt = Point(head.x + dx, head.y + dy)
            if not game.is_collision(pt):
                free_neighbors += 1

        # Normalize to [0..1] and give small reward for more free cells
        reward += 0.03 * (free_neighbors / 4.0)

        game.debug_info["free_neighbors"] = free_neighbors

        # --- 5) Wall proximity penalty (small, normalized) ---
        dist_to_wall = min(
            head.x,
            game.w - BLOCK_SIZE - head.x,
            head.y,
            game.h - BLOCK_SIZE - head.y
        )

        # Normalize in blocks
        dist_blocks = dist_to_wall / BLOCK_SIZE

        # If very near wall (< 1 block) strong-ish penalty, otherwise tiny
        reward -= 0.05 * (1.0 if dist_blocks <=
                          1 else (1.0 / (dist_blocks + 1.0)))

        # --- 6) Tail proximity penalty (normalized) ---
        tail = game.snake[-1]
        manhattan_tail_blocks = manhattan_distance_blocks(head, tail)

        tail_pen = 0.0
        if manhattan_tail_blocks < 4:  # if tail is close (in grid units)
            tail_pen = 0.08 * (1.0 - (manhattan_tail_blocks / 4.0))
            reward -= tail_pen

        # --- 7) Distance-to-food shaping (normalized, small, clipped) ---
        # Use Manhattan distance in grid cells normalized by board perimeter (grid units)
        grid_w = game.w / BLOCK_SIZE
        grid_h = game.h / BLOCK_SIZE
        max_grid = grid_w + grid_h
        food = min(game.foods, key=lambda f: abs(
            f.x - head.x) + abs(f.y - head.y))
        
        dist_old = manhattan_distance_blocks(food, game.old_head)
        dist_new = manhattan_distance_blocks(food, head)

        delta = (dist_old - dist_new) / max_grid   # small number
        # apply small scaling and clip
        reward += float(np.clip(delta * 1.0, -0.3, 0.3))

        # --- 8) Circular motion penalty (gentle) ---
        reward += self.circular_motion_penalty() * 0.6  # scale down previous behavior

        # Clip total reward so a single step doesn't exceed reasonable bounds
        reward = float(np.clip(reward, -10.0, 10.0))

        return reward

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update Target Network (Soft Update)
        self.update_target_network()

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        final_move = [0, 0, 0]

        if random.random() < self.epsilon:
            move = random.randint(0, len(final_move) - 1)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

            self.last_prediction = prediction.detach().numpy()
            self.last_action = move

        final_move[move] = 1

        return final_move

    @property
    def last_loss(self):
        return self.trainer.last_loss
