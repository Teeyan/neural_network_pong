from environment import Environment
from q_learn_agent import QAgent
from sarsa_agent import SARSAAgent
from behavioral_cloning import BCAgent


def main():

    bc_agent = BCAgent()
    bc_agent.load_parameters()
    #train_acc = bc_agent.evaluate_on_train()
    #print(train_acc)
    #bc_agent.train_agent(300, 128)
    avg_score = bc_agent.test_agent(200)
    train_acc = bc_agent.evaluate_on_train()
    print(train_acc)
    print("Agent Averaged : " + str(avg_score) + " Bounces Over " + str(200) + "games.")
    return

    game_environ = Environment()
    # sarsa_agent = SARSAAgent()
    q_agent = QAgent()
    train_games = 100000
    test_games = 200

    # Train the agent for N games
    # sarsa_agent.train_agent(train_games)
    q_agent.train_agent(train_games)

    # Test the agent for N' games
    # avg_score = sarsa_agent.test_agent(test_games)
    avg_score = q_agent.test_agent(test_games)

    print("Agent Averaged : " + str(avg_score) + " Bounces Over " + str(test_games) + " games.")

if __name__ == "__main__":
    main()
