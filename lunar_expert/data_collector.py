from lunar_train import run_episode
import torch
from network import MLP
from network import DQNAgent
import gymnasium as gym
import minari

if __name__ == "__main__":

    num_eval_episodes = 20
    eval_cycle = 50

    env = gym.make("LunarLander-v3",continuous=False,gravity=-9.8,enable_wind=False,
                   wind_power=15.0,turbulence_power=1.5)
    
    env = minari.DataCollector(env)
    env.reset()
    
    device = torch.device("cuda" if torch.cuda.is_available() else"cpu")
    
    print(f"using device: {device}")
    #print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    modelFile = "./models/dqn_agent_expert.pt"
    # TODO: Define Q network, target network and DQN agent
    Q = MLP(state_dim=8,action_dim=4,device=device)
    Q_target = MLP(state_dim=8,action_dim=4,device=device)
    
    agent = DQNAgent(Q,Q_target,4,device=device)
    agent.load(modelFile)
    
    print("model_loaded")


    for i in range(10):
        run_episode(env,agent,True,False,False)
        
    dataset = env.create_dataset("lunarLander/reverseCurriculum-v0",algorithm_name="DQN-1")