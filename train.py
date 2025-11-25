# train.py
# PPO Super Mario Bros — run logging + param change log intégré
import os, time, json, shutil, argparse
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.multiprocessing as _mp

from src.env import MultipleEnvironments
from src.model import PPO
from src.process import eval as eval_process


# ---------- Utils logging ----------
def _safe_mkdir(p):
    os.makedirs(p, exist_ok=True)
    return p

def _timestamp():
    return time.strftime("%Y%m%d-%H%M%S")

def _init_run_dirs(opt):
    # Un sous-dossier par run pour éviter d’écraser
    run_name = f"{_timestamp()}_w{opt.world}s{opt.stage}_{opt.action_type}"
    if opt.run_id:
        run_name += f"_{opt.run_id}"
    log_dir   = _safe_mkdir(os.path.join(opt.log_path, run_name))
    save_dir  = _safe_mkdir(os.path.join(opt.saved_path, run_name))
    return run_name, log_dir, save_dir

def _dump_run_info(log_dir, opt, envs):
    info = {
        "ts": int(time.time()),
        "world": opt.world,
        "stage": opt.stage,
        "action_type": opt.action_type,
        "lr": opt.lr,
        "gamma": opt.gamma,
        "tau": opt.tau,
        "beta": opt.beta,
        "epsilon": opt.epsilon,
        "batch_size": opt.batch_size,
        "num_epochs": opt.num_epochs,
        "num_local_steps": opt.num_local_steps,
        "num_processes": opt.num_processes,
        "save_interval": opt.save_interval,
        "max_actions": opt.max_actions,
        "note": opt.note,
        "num_states": envs.num_states,
        "num_actions": envs.num_actions,
        "torch_cuda": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }
    with open(os.path.join(log_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)

def _append_param_change(log_dir, change, details, reason):
    line = {
        "ts": int(time.time()),
        "change": change,    # "initial" | "update" | "tuned" ...
        "details": details,  # dict des params changés/valeurs
        "reason": reason     # string courte
    }
    with open(os.path.join(log_dir, "param_changes.jsonl"), "a") as f:
        f.write(json.dumps(line) + "\n")

def _maybe_init_tb(log_dir):
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=log_dir)
    except Exception:
        return None


# ---------- CLI ----------
def get_args():
    p = argparse.ArgumentParser("PPO Mario — minimal but logged")
    p.add_argument("--world", type=int, default=1)
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--action_type", type=str, default="simple", choices=["right", "simple", "complex"])

    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--gamma', type=float, default=0.9)   
    p.add_argument('--tau', type=float, default=1.0)     
    p.add_argument('--beta', type=float, default=0.01)   
    p.add_argument('--epsilon', type=float, default=0.2) 

    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_epochs', type=int, default=10)
    p.add_argument("--num_local_steps", type=int, default=512)
    p.add_argument("--num_global_steps", type=int, default=int(5e6))
    p.add_argument("--num_processes", type=int, default=8)

    p.add_argument("--save_interval", type=int, default=50) # save every N episodes
    p.add_argument("--max_actions", type=int, default=200)  # eval loop cap

    p.add_argument("--log_path", type=str, default="tensorboard/ppo_super_mario_bros")
    p.add_argument("--saved_path", type=str, default="trained_models")

    p.add_argument("--note", type=str, default="", help="short reason / context for the run")
    p.add_argument("--run_id", type=str, default="", help="tag suffix for folder names")

    return p.parse_args()


def train(opt):
    run_name, log_dir, save_dir = _init_run_dirs(opt)

    writer = _maybe_init_tb(log_dir)

    _append_param_change(
        log_dir,
        change="initial",
        details={
            "lr": opt.lr, "gamma": opt.gamma, "tau": opt.tau, "beta": opt.beta,
            "epsilon": opt.epsilon, "batch_size": opt.batch_size,
            "num_epochs": opt.num_epochs, "num_local_steps": opt.num_local_steps,
            "num_processes": opt.num_processes
        },
        reason=f"baseline run{(' — ' + opt.note) if opt.note else ''}"
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    _safe_mkdir(opt.log_path)
    _safe_mkdir(opt.saved_path)

    mp = _mp.get_context("spawn")
    envs = MultipleEnvironments(opt.world, opt.stage, opt.action_type, opt.num_processes)
    _dump_run_info(log_dir, opt, envs)

    model = PPO(envs.num_states, envs.num_actions)
    if torch.cuda.is_available():
        model.cuda()
    model.share_memory()

    proc_eval = mp.Process(target=eval_process, args=(opt, model, envs.num_states, envs.num_actions))
    proc_eval.daemon = True
    proc_eval.start()

    
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

   
    [agent_conn.send(("reset", None)) for agent_conn in envs.agent_conns]
    curr_states = [agent_conn.recv() for agent_conn in envs.agent_conns]
    curr_states = torch.from_numpy(np.concatenate(curr_states, 0))
    if torch.cuda.is_available():
        curr_states = curr_states.cuda()

    curr_episode = 0
    steps_global = 0

    while True:
        curr_episode += 1

        old_log_policies = []
        actions = []
        values = []
        states = []
        rewards = []
        dones = []

        
        ep_reward_sum = 0.0
        for _ in range(opt.num_local_steps):
            states.append(curr_states)
            logits, value = model(curr_states)
            values.append(value.squeeze())
            policy = F.softmax(logits, dim=1)
            old_m = Categorical(policy)
            action = old_m.sample()

            actions.append(action)
            old_log_policies.append(old_m.log_prob(action))

            
            if torch.cuda.is_available():
                [c.send(("step", a)) for c, a in zip(envs.agent_conns, action.cpu())]
            else:
                [c.send(("step", a)) for c, a in zip(envs.agent_conns, action)]

            state, reward, done, info = zip(*[c.recv() for c in envs.agent_conns])
            state = torch.from_numpy(np.concatenate(state, 0))

            
            if torch.cuda.is_available():
                state  = state.cuda()
                reward = torch.cuda.FloatTensor(reward)
                done   = torch.cuda.FloatTensor(done)
            else:
                reward = torch.FloatTensor(reward)
                done   = torch.FloatTensor(done)

            rewards.append(reward)
            dones.append(done)
            curr_states = state

            # log reward (mean across envs)
            ep_reward_sum += float(reward.mean().item())
            steps_global += len(envs.agent_conns)

            if steps_global >= opt.num_global_steps:
                break

        if steps_global >= opt.num_global_steps:
            print("Reached num_global_steps — stopping.")
            break

        # -------- GAE / returns --------
        _, next_value = model(curr_states)
        next_value = next_value.squeeze()

        old_log_policies = torch.cat(old_log_policies).detach()
        actions = torch.cat(actions)
        values = torch.cat(values).detach()
        states = torch.cat(states)

        gae = 0
        R = []
        # reverse loop
        for value, reward, done in list(zip(values, rewards, dones))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() * (1 - done) - value.detach()
            next_value = value
            R.append(gae + value)
        R = torch.cat(R[::-1]).detach()
        advantages = R - values

        # -------- PPO updates --------
        total_loss_val = None
        for _ in range(opt.num_epochs):
            idx = torch.randperm(opt.num_local_steps * opt.num_processes)
            
            for j in range(opt.batch_size):
                b = int(opt.num_local_steps * opt.num_processes / opt.batch_size)
                batch_idx = idx[j * b:(j + 1) * b]
                logits, value = model(states[batch_idx])
                new_policy = F.softmax(logits, dim=1)
                new_m = Categorical(new_policy)
                new_log_policy = new_m.log_prob(actions[batch_idx])

                ratio = torch.exp(new_log_policy - old_log_policies[batch_idx])
                surr1 = ratio * advantages[batch_idx]
                surr2 = torch.clamp(ratio, 1.0 - opt.epsilon, 1.0 + opt.epsilon) * advantages[batch_idx]
                actor_loss = -torch.mean(torch.min(surr1, surr2))
                critic_loss = F.smooth_l1_loss(R[batch_idx], value.squeeze())
                entropy_loss = torch.mean(new_m.entropy())
                total_loss = actor_loss + critic_loss - opt.beta * entropy_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                total_loss_val = float(total_loss.item())

        # -------- logs / saves --------
        if writer:
            writer.add_scalar("train/total_loss", total_loss_val, curr_episode)
            writer.add_scalar("train/mean_reward_per_rollout",
                              ep_reward_sum / float(opt.num_local_steps),
                              curr_episode)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], curr_episode)

        if curr_episode % opt.save_interval == 0:
            path_latest = os.path.join(save_dir, f"ppo_latest.pt")
            path_ep     = os.path.join(save_dir, f"ppo_ep{curr_episode}.pt")
            torch.save(model.state_dict(), path_latest)
            torch.save(model.state_dict(), path_ep)
            print(f"[save] {path_ep}")

        print(f"Episode: {curr_episode}. Total loss: {total_loss_val:.6f}")

    # cleanup
    if writer:
        writer.close()


if __name__ == "__main__":
    opt = get_args()
    # HEADLESS par défaut (WSL), process d’éval ne fera pas env.render()
    os.environ.setdefault("HEADLESS", "1")
    train(opt)
