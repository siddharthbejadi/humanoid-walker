"""Counted-step budget checks for fresh replay after checkpoint resume.

Checkpoints restore weights/optimizer, not replay. Parallel worlds increase the
collection needed for a temporal training sequence; aggregate FPS is not proof
that an optimizer update happened.
"""


def validate_learning_budget(worlds, batch_length, action_repeat, capacity, remaining):
    if min(worlds, batch_length, action_repeat, capacity, remaining) <= 0:
        raise ValueError("Learning budget values must be positive")
    # OnlineTrainer requires count/worlds > batch_length+1. Allow one more
    # collection for pretrain=0 and the Every scheduler's first optimizer call.
    frames = batch_length + 2
    if capacity // worlds < frames:
        raise ValueError(f"Replay needs at least {frames} frames per world; "
                         f"capacity={capacity}, worlds={worlds}")
    conservative_warmup = (frames + 1) * worlds * action_repeat
    if remaining <= conservative_warmup:
        raise ValueError(f"Remaining budget {remaining} does not exceed conservative "
                         f"fresh-replay warm-up {conservative_warmup}; reduce worlds "
                         "or explicitly increase the authorized budget")
    return conservative_warmup
