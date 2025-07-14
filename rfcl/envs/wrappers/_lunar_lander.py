from rfcl.envs.wrappers.curriculum import InitialStateWrapper
from Box2D import b2Vec2
#Debug to see if it works appropriately
class LunarLanderInitialStateWrapper(InitialStateWrapper):
    def set_env_state(self, state):
    
        ###HARD CODE THESE VALUES OUTSIDE FUNC
        kx = (600/30.0/2)
        ky = (400/30.0/2)
        jy = (400/30.0/4 + 18/30.0)
        FPS = 50
        kvx = 600/30.0/2
        kvy = 400/30.0/2

        x = float(kx*(state[0] + 1))
        y = float(ky*state[1] + jy)
        vx = float(state[2]*FPS/kvx)
        vy = float(state[3]*FPS/kvy)
        w = float(state[5] * FPS/20)

        c1 = state[6] == 1.0
        c2 = state[7] == 1.0

        self.unwrapped.lander.position = b2Vec2(x,y)
        self.unwrapped.lander.angle = float(state[4])
        self.unwrapped.lander.angularVelocity = float(w)
        self.unwrapped.lander.linearVelocity = b2Vec2(vx,vy)
        self.unwrapped.legs[0].ground_contact = c1
        self.unwrapped.legs[1].ground_contact = c2

        #makes sure legs also reset to correct position
        #Otherwise sometimes start spinning around
        self.unwrapped.legs[0].position=(x, y)
        self.unwrapped.legs[1].position=(x - 2/3, y)

    def get_env_obs(self):# reimplement, shouldn't just step!
        obs = self.env.unwrapped.step(0)
        return obs[0]