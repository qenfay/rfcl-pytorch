import gymnasium as gym
from Box2D import b2Vec2


def transform(state,env):
    
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

    env.unwrapped.lander.position = b2Vec2(x,y)
    env.unwrapped.lander.linearVelocity = b2Vec2(vx,vy)
    env.unwrapped.lander.angle = float(state[4])
    env.unwrapped.lander.angularVelocity = float(w)
    env.unwrapped.legs[0].ground_contact = c1
    env.unwrapped.legs[1].ground_contact = c2
    
    #makes sure legs also reset to correct position
    #Otherwise sometimes start spinning around
    env.unwrapped.legs[0].position=(x, y)
    env.unwrapped.legs[1].position=(x -  2/3, y)


env = gym.make("LunarLander-v2",render_mode = "human")
state = env.reset()[0]



saved_state = state
step = 1

while True:
    
    if step % 40 == 0:
        transform(saved_state,env)
        print("set state!")
        saved_state = state
        
    state, reward, terminal,truncated,info = env.step(0)
    env.render()
    
    if terminal:
        saved_state = env.reset()[0]
        print("reset")
        step = 1
        
    step+=1
    