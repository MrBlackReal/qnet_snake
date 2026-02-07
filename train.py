# train.py
from agent import Agent
from snake_game import SnakeGameAI
from collections import deque
import numpy as np
import torch
import os

# Set number of threads to available CPU cores for faster calculation
# Default to 4 if cpu_count is unavailable
cpu_count = os.cpu_count()
torch.set_num_threads((cpu_count - 1 if cpu_count > 1 else 1))

print(f"CPU count:", cpu_count)

def train():
    record = 0
    score_window = deque(maxlen=100)
    loss_window = deque(maxlen=100)

    stagnation_counter = 0
    last_best_score = 0

    agent = Agent()
    game = SnakeGameAI(num_food=10)

    agent.model.load("best_model.pth")

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)

        # Perform step and get game flags
        done, score, food_eaten, changed_direction = game.play_step(final_move)

        # Calculate reward in one place
        reward = agent.compute_reward(
            game, done, food_eaten, changed_direction)

        state_new = agent.get_state(game)

        # Update Debug Info
        game.debug_info = {
            'epsilon': agent.epsilon,
            'reward': reward,
            'q_values': agent.last_prediction if agent.last_prediction is not None else [0, 0, 0],
            'action': np.argmax(final_move),
            'state': state_old,
            'loss': agent.last_loss,
            "score": score,
            "high_score": record,
            "free_neighbors": 0
        }

        # Training
        agent.train_short_memory(
            state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        if food_eaten:
            agent.train_long_memory()

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            score_window.append(score)
            avg_score = sum(score_window) / len(score_window)

            loss_window.append(game.debug_info["loss"])
            avg_loss = sum(loss_window) / len(loss_window)

            if record < score:
                record = score
                agent.train_long_memory()

                game.debug_info["high_score"] = record

                stagnation_counter = 0
                agent.model.save(f"best_model.pth")
            else:
                stagnation_counter += 1

            # If we haven't beaten the record in 80 games, spike entropy
            if stagnation_counter > 75:
                # Force random moves to break loops, but max 1.0
                agent.epsilon = 0.01 + 0.02 * np.exp(-agent.n_games / 500)
                print(
                    f"Stagnation detected: Boosting exploration to {agent.epsilon:.2f}!")
                stagnation_counter = 0

            if agent.n_games % 100 == 0:
                agent.model.save(f"{agent.n_games}_model.pth")

            print(f"Game {agent.n_games} | Score {score}/{record} | Avg100 {avg_score:.4f} | AvgLoss100 {avg_loss:.3f} | Stagnation {stagnation_counter} | Eps {agent.epsilon:.3f}")


if __name__ == "__main__":
    train()
