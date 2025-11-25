from gym_super_mario_bros import make as make_mario
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from compat import reset_compat, step_compat

env = make_mario('SuperMarioBros-1-1-v3', apply_api_compatibility=True)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

obs, _ = reset_compat(env)
print("READY:", env.observation_space, env.action_space)

steps = 0
while steps < 50:
    a = env.action_space.sample()
    obs, r, done, info = step_compat(env, a)
    steps += 1
    if done:
        obs, _ = reset_compat(env)

env.close()
print(f"OK — {steps} steps")
