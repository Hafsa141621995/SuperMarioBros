
import os, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
import imageio.v2 as imageio

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

from compat import reset_compat, step_compat
from src.model import PPO  # PPO(num_states, num_actions) positional args

def now_tag(): return time.strftime("%Y%m%d-%H%M%S")

def pick_actions(kind: str):
    kind = kind.lower()
    if kind in ("right", "right_only"): return RIGHT_ONLY
    if kind in ("complex",):            return COMPLEX_MOVEMENT
    return SIMPLE_MOVEMENT

def make_env(world: int, stage: int, actions, seed: int = 123):
  
    env_step = gym_super_mario_bros.make(
        f"SuperMarioBros-{world}-{stage}-v3",
        apply_api_compatibility=True,
    )
    env_step = JoypadSpace(env_step, actions)

    
    env_render = gym_super_mario_bros.make(
        f"SuperMarioBros-{world}-{stage}-v3",
        render_mode="rgb_array",
        apply_api_compatibility=True,
    )
    env_render = JoypadSpace(env_render, actions)


    try:
        env_step.reset(seed=seed)
    except TypeError:
        pass
    try:
        env_render.reset(seed=seed)
    except TypeError:
        pass
    return env_step, env_render

def to_uint8_rgb(img):
 
    if img is None:
        raise RuntimeError("render() returned None")
    if img.ndim == 2:  
        img = np.stack([img]*3, axis=-1)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8, copy=False)
    return np.ascontiguousarray(img)

def get_rgb_frame(render_env):
  
    try:
        img = render_env.render()
    except TypeError:
       
        img = render_env.render("rgb_array")
    return to_uint8_rgb(img)

@torch.no_grad()
def run_eval(ckpt: str, world: int, stage: int, action_type: str,
             outdir: str, steps: int, fps: int, gif_stride: int):
    os.makedirs(outdir, exist_ok=True)
    actions = pick_actions(action_type)
    num_actions = len(actions)

    # model
    num_states = 4  # you trained with 4 stacked frames
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PPO(num_states, num_actions).to(device)
    state_dict = torch.load(ckpt, map_location=device)
    model.load_state_dict(state_dict); model.eval()

 
    seed = 123
    env_step, env_render = make_env(world, stage, actions, seed=seed)

    obs, _ = reset_compat(env_step)
    obs = np.ascontiguousarray(obs).copy()

    
    if obs.ndim == 3 and obs.shape[-1] == 4:
        obs = np.transpose(obs, (2, 0, 1)).copy()
    s = torch.from_numpy(obs[None, ...]).float().to(device) 


    try:
        reset_compat(env_render)
    except Exception:
        pass

    tag = now_tag()
    mp4_path = os.path.join(outdir, f"w{world}s{stage}_ppo_eval_{tag}.mp4")
    gif_path = os.path.join(outdir, f"w{world}s{stage}_ppo_eval_{tag}.gif")
    writer = imageio.get_writer(mp4_path, fps=fps, codec="libx264", quality=8)
    gif_frames = []

    total_reward = 0.0
    for t in range(steps):
        logits, _ = model(s)
        action = torch.argmax(F.softmax(logits, dim=1), dim=1).item()

        obs_next, r, done, info = step_compat(env_step, action)
        total_reward += float(r)

        
        obs_next = np.ascontiguousarray(obs_next).copy()
        if obs_next.ndim == 3 and obs_next.shape[-1] == 4:
            obs_next = np.transpose(obs_next, (2, 0, 1)).copy()
        s = torch.from_numpy(obs_next[None, ...]).float().to(device)

        frame = get_rgb_frame(env_render)
        writer.append_data(frame)
        if gif_stride > 0 and (t % gif_stride == 0):
            gif_frames.append(frame)

        if done:
            break

    writer.close()
    if gif_frames:
        imageio.mimsave(gif_path, gif_frames, fps=max(1, fps // max(1, gif_stride)))


    best_gif = os.path.join(outdir, "mario_best.gif")
    best_mp4 = os.path.join(outdir, "mario_best.mp4")
    if os.path.exists(mp4_path): 
        try: os.replace(mp4_path, best_mp4)
        except: pass
    if gif_frames and os.path.exists(gif_path):
        try: os.replace(gif_path, best_gif)
        except: pass

    print(f"score={total_reward:.3f}")
    print(f"(mp4 saved) {best_mp4}")
    if gif_frames: print(f"(gif saved) {best_gif}")
    return best_mp4, (best_gif if gif_frames else None), total_reward

def main():
    ap = argparse.ArgumentParser("Mario eval (clean RGB → MP4+GIF)")
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--action_type", type=str, default="simple",
                    choices=["right", "simple", "complex"])
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="output_eval")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--gif_stride", type=int, default=8)
    args = ap.parse_args()
    run_eval(args.checkpoint, args.world, args.stage,
             args.action_type, args.outdir, args.steps, args.fps, args.gif_stride)

if __name__ == "__main__":
    main()
