import copy
import os
import os.path as osp
import sys
import warnings
from dataclasses import asdict, dataclass
from typing import Optional

import gymnasium as gym
import jax
import numpy as np
import optax
import minari
#from omegaconf import OmegaConf

from rfcl.agents.sac import SAC, ActorCritic, SACConfig
from rfcl.agents.sac.networks import DiagGaussianActor
from rfcl.data.dataset import ReplayDataset, get_states_dataset
from rfcl.envs.make_env import EnvConfig, get_initial_state_wrapper, make_env_from_cfg
from rfcl.envs.wrappers.curriculum import ReverseCurriculumWrapper
from rfcl.envs.wrappers.forward_curriculum import SeedBasedForwardCurriculumWrapper
from rfcl.logger import LoggerConfig
from rfcl.models import NetworkConfig, build_network_from_cfg
from rfcl.utils.parse import parse_cfg
from rfcl.utils.spaces import get_action_dim

from dacite import from_dict


@dataclass
class TrainConfig:
    steps: int
    actor_lr: float
    critic_lr: float
    dataset_path: str
    shuffle_demos: bool
    num_demos: int

    data_action_scale: Optional[float]

    # reverse curriculum wrapper configs
    reverse_step_size: int
    curriculum_method: str
    start_step_sampler: str
    per_demo_buffer_size: int
    demo_horizon_to_max_steps_ratio: float
    train_on_demo_actions: bool

    # forward curriculum configs
    forward_curriculum: str
    staleness_transform: str
    staleness_coef: float
    staleness_temperature: float
    score_transform: str
    score_temperature: float
    num_seeds: int

    # stage 2 training configs
    load_actor: bool
    load_critic: bool
    load_as_offline_buffer: bool
    load_as_online_buffer: bool

    # other configs that are generally used for experimentation
    use_orig_env_for_eval: bool = True
    eval_start_of_demos: bool = False


@dataclass
class SACNetworkConfig:
    actor: NetworkConfig
    critic: NetworkConfig


@dataclass
class SACExperiment:
    seed: int
    sac: SACConfig
    env: EnvConfig
    eval_env: EnvConfig
    train: TrainConfig
    network: SACNetworkConfig
    logger: Optional[LoggerConfig]
    verbose: int
    algo: str = "sac"
    stage_1_model_path: str = None  # if not None, will load pretrained stage 1 model and skip to stage 2 of training
    save_eval_video: bool = True  # whether to save eval videos
    stage_1_only: bool = False  # stop training after reverse curriculum completes
    stage_2_only: bool = False # skip stage 1 training
    demo_seed: int = None  # fix a seed to fix which demonstrations are sampled from a dataset


#Load minari dataset into compatible strucutre, add the extra features from
#get_states_dataset()
def load_minari_dataset(dataset_name):
    
    dataset = minari.load_dataset(dataset_name)
    filter_dataset = dataset.filter_episodes(lambda episode: episode.rewards.sum() > 180)

    i = 0
    out = {}
    for episode in filter_dataset:
        out[i] = dict(state=episode.observations,seed=0,reset_kwargs={},demo_id=i)
        i+=1
        
    return out

        
    
    
def main(cfg:SACExperiment):
    np.random.seed(cfg.seed) #set seed ig
    
    cfg.sac.num_envs = cfg.env.num_envs
    cfg.sac.num_eval_envs = cfg.eval_env.num_envs #Not sure why these 2 done
    
    #First function to be called in their code
    #Hardcode for now
    #change to this get_states_dataset(cfg.train.dataset_path, num_demos=cfg.train.num_demos)
    #dataset = get_states_dataset(demo_dataset_path="~/.minari/datasets/lunarLander/reverseCurriculum-v0/data/main_data.h5",num_demos=3)

    states_dataset = load_minari_dataset("lunarLander/reverseCurriculum-v0")
    
    #print(states_dataset)
    
    #Hardcoded for now, update later
    reward_mode = "dense" 
    
    #Skipping the optional "train_on_demo_actions",
    #Implement it later
    
    #Create Env Wrapper, Have to modify files for this :(
        
    InitialStateWrapper = get_initial_state_wrapper(cfg.env.env_id)
    
    #copied as is
    wrappers = [
        lambda env: InitialStateWrapper(
        env,
        states_dataset=states_dataset,
        demo_horizon_to_max_steps_ratio=cfg.train.demo_horizon_to_max_steps_ratio,
    )
]
    ##env is created, some changes into their codebase, should refactor later
    env, env_meta = make_env_from_cfg(cfg.env, seed=cfg.seed, wrappers=wrappers)
    eval_env = None
    use_orig_env_for_eval = cfg.train.use_orig_env_for_eval    
    link_envs = []
    if not use_orig_env_for_eval:
        eval_env = ReverseCurriculumWrapper(
            eval_env,
            eval_mode=True,
            eval_start_of_demos=cfg.train.eval_start_of_demos,
            states_dataset=states_dataset,
            reverse_step_size=cfg.train.reverse_step_size,
            curriculum_method=cfg.train.curriculum_method,
            per_demo_buffer_size=cfg.train.per_demo_buffer_size,
            start_step_sampler=cfg.train.start_step_sampler,
        )
        link_envs = [eval_env]
    env = ReverseCurriculumWrapper(
        env,
        states_dataset=states_dataset,
        reverse_step_size=cfg.train.reverse_step_size,
        curriculum_method=cfg.train.curriculum_method,
        per_demo_buffer_size=cfg.train.per_demo_buffer_size,
        start_step_sampler=cfg.train.start_step_sampler,
        link_envs=link_envs,
    )

    sample_obs, sample_acts = env_meta.sample_obs, env_meta.sample_acts

    sample_acts = sample_acts.reshape(-1,1)
    # create actor and critics models
    act_dims = get_action_dim(env_meta.act_space)
    
    def create_ac_model():
        actor = DiagGaussianActor(
            feature_extractor=build_network_from_cfg(cfg.network.actor),
            act_dims=act_dims,
            state_dependent_std=True,
        )
        ac = ActorCritic.create(
            jax.random.PRNGKey(cfg.seed),
            actor=actor,
            critic_feature_extractor=build_network_from_cfg(cfg.network.critic),
            sample_obs=sample_obs,
            sample_acts=sample_acts,
            initial_temperature=cfg.sac.initial_temperature,
            actor_optim=optax.adam(learning_rate=cfg.train.actor_lr),
            critic_optim=optax.adam(learning_rate=cfg.train.critic_lr),
        )
        return ac
    
    ac = create_ac_model()
    cfg.logger.cfg = asdict(cfg)
    logger_cfg = cfg.logger
    algo = SAC(
        env=env,
        eval_env=eval_env,
        env_type=cfg.env.env_type,
        ac=ac,
        logger_cfg=logger_cfg,
        cfg=cfg.sac,
    )
    
    ###Works till here ig!

if __name__ == "__main__":
   
    #dataset = minari.load_dataset("lunarLander/reverseCurriculum-v0")


    sac_cfg = SACConfig(
        num_seed_steps=5000,seed_with_policy=False,replay_buffer_capacity=200_000,batch_size=256,
        steps_per_env=1,grad_updates_per_step=80,actor_update_freq=20,num_qs=10,num_min_qs=2,
        discount=0.99, tau=0.005,backup_entropy=False,eval_freq=5_000,eval_steps=500,
        log_freq=1000,save_freq=10_000,learnable_temp=True,initial_temperature=1.0
    )
    
    env_cfg = EnvConfig(env_id="LunarLander-v2",env_type='gym:cpu',max_episode_steps=1000,
                        num_envs= 8, env_kwargs={}, action_scale=None)#GO over what args do
    
    eval_cfg = EnvConfig(env_id="LunarLander-v2",env_type='gym:cpu',max_episode_steps=1000,
                        num_envs= 2, env_kwargs={}, action_scale=None)
    
    train_cfg = TrainConfig(steps = 1_000_000,actor_lr= 3e-4,critic_lr=3e-4,
                            dataset_path="",shuffle_demos=True,num_demos=5,
                            data_action_scale=0,reverse_step_size=4,
                            start_step_sampler="geometric",curriculum_method="per_demo",
                            per_demo_buffer_size=3,demo_horizon_to_max_steps_ratio=3,
                            train_on_demo_actions=True,load_actor=True,load_critic=True,load_as_offline_buffer=True,
                            load_as_online_buffer=False,forward_curriculum="success_once_score",staleness_coef=0.1,
                            staleness_temperature=0.1,staleness_transform="rankmin",score_transform=0.1,
                            score_temperature=0.1,num_seeds=1000)
    
    network_cfg = SACNetworkConfig(
        actor=NetworkConfig(type="mlp",arch_cfg={"features":[256,256,256],"output_activation":"relu","use_layer_norm":False}),
        critic=NetworkConfig(type="mlp",arch_cfg={"features":[256,256,256],"output_activation":"relu",
                                                  "use_layer_norm":True})
        )###Make sure arch_cfg is correct type
    
    logger_cfg = LoggerConfig(workspace="exps")#GO read what this does
    
    
    cfg = SACExperiment(seed=0,sac = sac_cfg,env=env_cfg,eval_env=eval_cfg,
                        train=train_cfg,network=network_cfg,logger=logger_cfg,verbose=1) 
    
    #Compiles so far, need to add the required env files to make it work now!

    main(cfg)
    
