import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque
from snake_game import Direction, Point, BLOCK_SIZE, W_WIDTH, W_HEIGHT
from net import Conv_QNet
from utils import manhattan_distance, manhattan_distance_blocks

from config import MAX_MEMORY, BATCH_SIZE, LR

class QTrainerCNN:
    def __init__(self, model, target_model, lr, gamma):
        self.model = model
        self.target_model = target_model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.last_loss = 0.0

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        # Handle single-sample inputs (when training short memory)
        # State shape for CNN is (3, H, W), so len == 3 means single sample
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        pred = self.model(state)
        target = pred.clone().detach()

        # Predict next state Q-values for the whole batch at once to avoid shape errors
        # and improve performance.
        with torch.no_grad():
            next_pred = self.target_model(next_state)

        for i in range(len(done)):
            q_new = reward[i]
            if not done[i]:
                # Double DQN / Target Network approach
                q_new = reward[i] + self.gamma * torch.max(next_pred[i])

            target[i][torch.argmax(action[i]).item()] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        self.last_loss = loss.item()


class CNNAgent:
    def __init__(self, w_width=W_WIDTH, w_height=W_HEIGHT, epsilon=1, gamma=0.99):
        self.n_games = 0
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.gamma = gamma
        
        # Grid dimensions for the CNN
        self.board_w = w_width // BLOCK_SIZE
        self.board_h = w_height // BLOCK_SIZE

        self.memory = deque(maxlen=MAX_MEMORY)
        
        # Initialize CNN Models
        # Output size 3 (Straight, Right, Left) or 4 (Up, Down, Left, Right)?
        # The original Agent uses relative directions [Straight, Right, Left]
        self.model = Conv_QNet(self.board_w, self.board_h, output_size=3)
        self.target_model = Conv_QNet(self.board_w, self.board_h, output_size=3)
        
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.trainer = QTrainerCNN(self.model, self.target_model, lr=LR, gamma=self.gamma)
        
        self.head_history = deque(maxlen=4)
        
        self.last_prediction = None
        self.last_action = None

    def update_target_network(self, tau=0.01):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            
    def circular_motion_penalty(self):
        if len(self.head_history) < 4:
            return 0.0

        p0, p1, p2, p3 = self.head_history
        v1 = np.array([p1.x - p0.x, p1.y - p0.y], dtype=float)
        v2 = np.array([p2.x - p1.x, p2.y - p1.y], dtype=float)
        v3 = np.array([p3.x - p2.x, p3.y - p2.y], dtype=float)

        limit = 1e-6
        if np.linalg.norm(v1) < limit or np.linalg.norm(v2) < limit or np.linalg.norm(v3) < limit:
            return 0.0

        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        v3 /= np.linalg.norm(v3)

        turn1 = 1.0 - np.dot(v1, v2)
        turn2 = 1.0 - np.dot(v2, v3)

        if turn1 > 0.5 and turn2 > 0.5:
            return -0.08

        straightness = (1.0 - turn1) * (1.0 - turn2)
        return 0.02 * straightness

    def compute_reward(self, game, done, food_eaten, changed_direction):
        if done:
            return -20.0

        reward = 0.0
        head = game.snake[0]

        if food_eaten:
            reward += 8.0

        reward += 0.01

        if changed_direction:
            reward -= 0.02
        else:
            reward += 0.01

        free_neighbors = 0
        
        for dx, dy in [(BLOCK_SIZE, 0), (-BLOCK_SIZE, 0), (0, BLOCK_SIZE), (0, -BLOCK_SIZE)]:
            pt = Point(head.x + dx, head.y + dy)
            if not game.is_collision(pt):
                free_neighbors += 1

        reward += 0.03 * (free_neighbors / 4.0)
        game.debug_info["free_neighbors"] = free_neighbors

        dist_to_wall = min(
            head.x,
            game.w - BLOCK_SIZE - head.x,
            head.y,
            game.h - BLOCK_SIZE - head.y
        )
        dist_blocks = dist_to_wall / BLOCK_SIZE
        reward -= 0.05 * (1.0 if dist_blocks <= 1 else (1.0 / (dist_blocks + 1.0)))

        tail = game.snake[-1]
        manhattan_tail_blocks = manhattan_distance_blocks(head, tail)
        if manhattan_tail_blocks < 4:
            reward -= 0.08 * (1.0 - (manhattan_tail_blocks / 4.0))

        grid_w = game.w / BLOCK_SIZE
        grid_h = game.h / BLOCK_SIZE
        max_grid = grid_w + grid_h
        food = min(game.foods, key=lambda f: abs(f.x - head.x) + abs(f.y - head.y))
        
        dist_old = manhattan_distance_blocks(food, game.old_head)
        dist_new = manhattan_distance_blocks(food, head)
        delta = (dist_old - dist_new) / max_grid
        reward += float(np.clip(delta * 1.0, -0.3, 0.3))

        reward += self.circular_motion_penalty() * 0.6

        return float(np.clip(reward, -10.0, 10.0))

    def get_state(self, game):
        """
        Converts the game state into a 3-channel grid image:
        Channel 0: Snake Body
        Channel 1: Food
        Channel 2: Snake Head
        """
        # Update history for reward calculation
        head = game.snake[0]
        self.head_history.append(head)
        while len(self.head_history) < 4:
            self.head_history.append(head)

        w_grid = game.w // BLOCK_SIZE
        h_grid = game.h // BLOCK_SIZE
        
        state = np.zeros((3, h_grid, w_grid), dtype=np.float32)
        
        # 1. Snake Body
        for pt in game.snake:
            x = int(pt.x // BLOCK_SIZE)
            y = int(pt.y // BLOCK_SIZE)
            if 0 <= x < w_grid and 0 <= y < h_grid:
                state[0, y, x] = 1.0
                
        # 2. Food
        for food in game.foods:
            x = int(food.x // BLOCK_SIZE)
            y = int(food.y // BLOCK_SIZE)
            if 0 <= x < w_grid and 0 <= y < h_grid:
                state[1, y, x] = 1.0
                
        # 3. Head
        head = game.snake[0]
        x = int(head.x // BLOCK_SIZE)
        y = int(head.y // BLOCK_SIZE)
        if 0 <= x < w_grid and 0 <= y < h_grid:
            state[2, y, x] = 1.0
            
        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.update_target_network()

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        final_move = [0, 0, 0]
        
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0) # Add batch dim
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            
            self.last_prediction = prediction.detach().numpy()[0] # Store for debug
            self.last_action = move

        final_move[move] = 1
        return final_move

    @property
    def last_loss(self):
        return self.trainer.last_loss
