import os, time, glob, argparse
import numpy as np
import torch, imageio, imageio_ffmpeg 

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

from src.env import create_train_env
from src.model import PPO


def pick_actions(kind):
    if kind == "right":  return RIGHT_ONLY
    if kind == "simple": return SIMPLE_MOVEMENT
    return COMPLEX_MOVEMENT


def list_checkpoints(run_dir):
    paths = glob.glob(os.path.join(run_dir, "ppo_ep*.pt"))
    def key(p):
        import re
        m = re.search(r"ppo_ep(\d+)\.pt$", p)
        return int(m.group(1)) if m else -1
    return sorted(paths, key=key)


def get_frame(env):
    # nes-py exige 'rgb_array'
    try:
        img = env.render(mode="rgb_array")
        return img
    except TypeError:
        # vieux gym qui ignore le kw
        try:
            return env.render()
        except Exception:
            pass
    # fallback très bas-niveau
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "_get_image"):
        return env.unwrapped._get_image()
    return None


def step_any(env, a):
    """Unifie Gym v0/v26: retourne (obs, reward, done, info)."""
    out = env.step(a)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, reward, done, info
    obs, reward, done, info = out
    return obs, reward, bool(done), info


@torch.no_grad()
def eval_stream(model, world, stage, actions, steps, mp4_path=None, fps=30, gif_stride=8):
    env = create_train_env(world, stage, actions)
   
    r_env = JoypadSpace(
        gym_super_mario_bros.make(f"SuperMarioBros-{world}-{stage}-v3", apply_api_compatibility=True),
        actions
    )

    
    s_train = torch.from_numpy(env.reset())
    _ = r_env.reset()

    model.eval()
    total = 0.0
    frames = []

   
    writer = None
    if mp4_path:
        writer = imageio.get_writer(mp4_path, fps=fps, codec="libx264", quality=8)

    for t in range(int(steps)):
        logits, _ = model(s_train)
        probs = torch.softmax(logits, dim=1)
        act = torch.argmax(probs, dim=1).item()

        
        s_next, r, done, _ = env.step(act)
        total += float(r)

    
        _, _, r_done, _ = step_any(r_env, act)

       
        if t % int(gif_stride) == 0:
            fr = get_frame(r_env)
            if fr is not None:
                if fr.ndim == 2:  # grayscale -> RGB
                    fr = np.repeat(fr[..., None], 3, axis=2)
                if fr.dtype != np.uint8:
                    fr = fr.astype(np.uint8)
                frames.append(fr)
                if writer:
                    writer.append_data(fr)

        
        if done or r_done:
            s_train = torch.from_numpy(env.reset())
            _ = r_env.reset()
        else:
            s_train = torch.from_numpy(s_next)

    if writer:
        writer.close()

    return total, frames


def save_gif(frames, out_gif, fps=30):
    if frames:
        imageio.mimsave(out_gif, frames, fps=fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--action_type", choices=["right","simple","complex"], default="simple")
    ap.add_argument("--best_from", type=str, default=None)
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--outdir", type=str, default="output_eval")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--gif_stride", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    actions = pick_actions(args.action_type)

    # liste de checkpoints
    if args.checkpoint:
        cks = [args.checkpoint]
    elif args.best_from:
        cks = list_checkpoints(args.best_from)
    else:
        raise SystemExit("Spécifie --checkpoint ou --best_from")

    if not cks:
        raise SystemExit("Aucun checkpoint trouvé.")

    best = (-1e9, None, None)  
    ts = time.strftime("%Y%m%d-%H%M%S")  

    for ck in cks:
        print(f"→ {os.path.basename(ck)} ...", flush=True)
        # charge modèle
        # taille état/action dépend de l'env, donc on crée un env vite fait
        tmp_env = create_train_env(args.world, args.stage, actions)
        num_states = tmp_env.observation_space.shape[0]
        num_actions = len(actions)
        tmp_env.close()

        model = PPO(num_states, num_actions)
        state = torch.load(ck, map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        base = f"w{args.world}s{args.stage}_{os.path.basename(ck).replace('.pt','')}_{ts}"
        out_mp4 = os.path.join(args.outdir, f"{base}.mp4")
        score, frames = eval_stream(
            model, args.world, args.stage, actions,
            steps=args.steps, mp4_path=out_mp4, fps=args.fps, gif_stride=args.gif_stride
        )
        print(f"  score={score:.3f}  (mp4 saved: {out_mp4})")
        if score > best[0]:
            best = (score, frames, ck)

    # sauvegarde GIF du meilleur
    score, frames, ck = best
    base = f"best_w{args.world}s{args.stage}_{os.path.basename(ck).replace('.pt','')}_{ts}"
    out_gif = os.path.join(args.outdir, f"{base}.gif")
    save_gif(frames, out_gif, fps=args.fps)
    print(f"[best] {os.path.basename(ck)}  score={score:.3f}")
    print(f"[saved] {out_gif}")


if __name__ == "__main__":
    main()
