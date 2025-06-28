import sys
sys.path.append("../")

import numpy as np
import gymnasium as gym
from network import MLP
from network import DQNAgent
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os

class Evaluation:

    def __init__(self, store_dir, name, stats=[]):
        """
        Creates placeholders for the statistics listed in stats to generate tensorboard summaries.
        e.g. stats = ["loss"]
        """
        self.folder_id = "%s-%s" % (name, datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.summary_writer = SummaryWriter(os.path.join(store_dir, self.folder_id))
        self.stats = stats

    def write_episode_data(self, episode, eval_dict):
        """
        Write episode statistics in eval_dict to tensorboard, make sure that the entries in eval_dict are specified in stats.
        e.g. eval_dict = {"loss" : 1e-4}
        """

        for k in eval_dict:
            assert k in self.stats
            #print(f"writing episode {episode}")
            self.summary_writer.add_scalar(k, eval_dict[k], global_step=episode)

        self.summary_writer.flush()

    def close_session(self):
        self.summary_writer.close()
        
        
        
class EpisodeStats:
    """
    This class tracks statistics like episode reward or action usage.
    """

    def __init__(self):
        self.episode_reward = 0
        self.actions_ids = []

    def step(self, reward, action_id):
        self.episode_reward += reward
        self.actions_ids.append(action_id)

    def get_action_usage(self, action_id):
        ids = np.array(self.actions_ids)
        return len(ids[ids == action_id]) / len(ids)
    
    
    
    
def run_episode(
    env,
    agent,
    deterministic,
    skip_frames=5,
    do_training=True,
    rendering=False,
    max_timesteps=1000,
    history_length=0,
    max_neg_threshold = 50,
    max_neg = 30
):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """
    
    stats = EpisodeStats()
    step = 0
    state = env.reset()
    state = state[0]

    while True:


        action_id = agent.act(state=state, deterministic=deterministic)
        next_state, reward, terminal, truncated,info = env.step(action_id)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state

        if rendering:
            env.render()

        if terminal or step > max_timesteps or truncated:
            break

        step += 1



    return stats



def train_online(
    env,
    agent,
    num_episodes,
    history_length=0,
    model_dir="./models_carracing",
    tensorboard_dir="./tensorboard",
):

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train agent")
    tensorboard = Evaluation(
        os.path.join(tensorboard_dir, "train"),name = "lunar_expert",
        stats=["episode_reward", "main", "left", "right","Eval"],
    )

    max_timesteps = 2000#update
    k = 0
    for i in range(num_episodes):
        print("epsiode %d" % i)

   #     Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)

        stats = run_episode(
            env,
            agent,
            max_timesteps=max_timesteps,
            deterministic=False,
            do_training=True,
            history_length=history_length
        )

        tensorboard.write_episode_data(
            i,
            eval_dict={
                "episode_reward": stats.episode_reward,
                "main": stats.get_action_usage(2),
                "left": stats.get_action_usage(1),
                "right": stats.get_action_usage(3),
            },
        )
        

        if i % eval_cycle == 0:
            total = 0
            for j in range(num_eval_episodes):
                stats = run_episode(env, agent, deterministic=True, do_training=False,history_length=history_length)
                total += stats.episode_reward
            tensorboard.write_episode_data(i,{"Eval":total/num_eval_episodes})

        # store model.
        if i % (eval_cycle*4) == 0 or (i >= num_episodes - 1):
            agent.save(os.path.join(model_dir,f"dqn_agent_{i}.pt"))##update name
            k += 1
    tensorboard.close_session()



if __name__ == "__main__":

    num_eval_episodes = 20
    eval_cycle = 50

    env = gym.make("LunarLander-v3",continuous=False,gravity=-9.8,
                   enable_wind=True,wind_power=15.0,turbulence_power=1.5)
    
    
    # TODO: Define Q network, target network and DQN agent
    Q = MLP(state_dim=8,action_dim=4)
    Q_target = MLP(state_dim=8,action_dim=4)
    
    agent = DQNAgent(Q,Q_target,4)

    train_online(
        env, agent, num_episodes=10000, history_length=3, model_dir="./models"
    )