def reset_compat(env):
    out = env.reset()
    return out if isinstance(out, tuple) else (out, {})

def step_compat(env, action):
    out = env.step(action)
    if len(out) == 5:  
        obs, rew, term, trunc, info = out
        done = term or trunc
    else:           
        obs, rew, done, info = out
    return obs, rew, done, info
