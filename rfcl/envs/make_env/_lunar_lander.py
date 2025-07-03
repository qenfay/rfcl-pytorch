import gymnasium as gym


def is_lunar_lander_env(env_id: str):

    return env_id == "LunarLander-v2"


def env_factory(env_id: str, idx: int, env_kwargs=dict(), record_video_path: str = None, wrappers=[], record_episode_kwargs=dict()):
    def _init():
        env = gym.make(env_id,continuous=True)# Add keyword args from dataset here, for now hardcoded
        for wrapper in wrappers:
            env = wrapper(env)
        if record_video_path is not None and (not record_episode_kwargs["record_single"] or idx == 0):
            env = gym.wrappers.RecordVideo(
                env,
                record_video_path,
                episode_trigger = lambda x: True,
                disable_logger = True
            )
        return env
 
    return _init