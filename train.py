from agent import Agent
from snake_game import SnakeGameAI


def train():
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    # Optional: Load a pre-trained model if it exists
    agent.model.load("score_21_games_129_model.pth")

    while True:
        # Get old state
        state_old = agent.get_state(game)

        # Get move
        final_move = agent.get_action(state_old)

        # Perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        game.debug_info = {
            "epsilon": agent.epsilon,
            "q_values": agent.last_prediction,
            "action": agent.last_action,
            "state": state_old,
            "reward": reward
        }

        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save(f"score_{score}_games_{agent.n_games}_model.pth")

            if agent.n_games % 100 == 0:
                agent.model.save("latest.pth")

            print(
                f"Game {agent.n_games}, Score: {score}, Record: {record}, Reward: {reward}"
            )


if __name__ == "__main__":
    train()
