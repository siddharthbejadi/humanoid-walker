"""Teacher-only environment factory: MJX-Warp on GPU, with no CPU fallback."""

import atexit

from .mjx_teacher_env import MJXTeacherEnv


def make_envs(config):
    if getattr(config, "backend", None) != "mjx_teacher":
        raise ValueError("This trimmed bundle supports only backend=mjx_teacher")
    train_envs = MJXTeacherEnv(config)
    try:
        eval_count = int(getattr(config, "eval_episode_num", 0))
        eval_envs = MJXTeacherEnv(config, eval_count, evaluation=True) if eval_count else None
    except Exception:
        train_envs.close()
        raise
    atexit.register(train_envs.close)
    if eval_envs is not None:
        atexit.register(eval_envs.close)
    return train_envs, eval_envs, train_envs.observation_space, train_envs.action_space
