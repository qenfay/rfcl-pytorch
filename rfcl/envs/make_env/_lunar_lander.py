import gymnasium as gym
import math



class lunarInfoWrapper(gym.Wrapper):
    
    def __init__(self,env):
        super().__init__(env)
        self.count = 0
    
    def step(self,action):
        
        next_obs, rew, terminated, truncated, info = super().step(action)
        
        # print(f" awake: {self.unwrapped.lander.awake}")
        # print(f"action: {action}")
        # print(f" lander pos: {self.unwrapped.lander.position}")
        # print(f" angle {self.unwrapped.lander.angle}")
        # print(f" ang velocity {self.unwrapped.lander.angularVelocity}")
        x,y,vx,vy,t,w,c1,c2 = next_obs
        # print(next_obs)
        # print(self.count)
        info["success"]= False
        
        
        if rew > 90: 
            #print(f"reward: {rew}")
            info["success"]= True
            
            
        # if not terminated:
        #     if (math.isclose(x, 0.0, abs_tol=5e-2) and
        #         math.isclose(y, 0.0, abs_tol=5e-2) and
        #         math.isclose(vx, 0.0, abs_tol=1e-2) and
        #         math.isclose(vy, 0.0, abs_tol=1e-2) and
        #         math.isclose(t, 0.0, abs_tol=5e-2) and
        #         math.isclose(w, 0.0, abs_tol=5e-2) and
        #         c1 == 1.0 and c2 == 1.0):
                
        #         self.count += 1
        #         rew+=30
        #         print("success 1")
        #     else:
        #         self.count = 0
            
        #     if self.count > 3: #maybe add checks for legs touching or done? Also check if this works? Might not be sctrictly necessary!
        #         info["success"] = True
        #         print("adding rew") 
        #         rew+=100 #adding some peudo rewards if the success state is reached but the lander is awake!

            
        return next_obs,rew,terminated,truncated,info
    
    
def is_lunar_lander_env(env_id: str):

    return env_id == "LunarLander-v2"


def env_factory(env_id: str, idx: int, env_kwargs=dict(), record_video_path: str = None, wrappers=[], record_episode_kwargs=dict()):
    def _init():
        env = gym.make(env_id,continuous=True,render_mode="rgb_array")# Add keyword args from dataset here, for now hardcoded
        env = lunarInfoWrapper(env)
        for wrapper in wrappers:
            env = wrapper(env)
        if record_video_path is not None and (not record_episode_kwargs["record_single"] or idx == 0):
            env = gym.wrappers.RecordVideo(
                env,
                record_video_path,
                disable_logger = True,
            )
        return env
 
    return _init