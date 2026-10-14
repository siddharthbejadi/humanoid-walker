import torch
from torch import distributions as torchd
from torch.nn import functional as F

from tools import to_f32, to_i32


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    def __init__(self, logits, unimix_ratio=0.0):
        # (..., K)
        probs = F.softmax(to_f32(logits), dim=-1)
        uniform = unimix_ratio / probs.shape[-1]
        probs = probs * (1.0 - unimix_ratio) + torch.ones_like(probs, dtype=torch.float32) * uniform
        logits = torch.log(probs)
        super().__init__(logits=logits)

    @property
    def mode(self):
        # (..., K)
        _mode = F.one_hot(torch.argmax(self.logits, axis=-1), self.logits.shape[-1])
        return _mode.detach() + self.logits - self.logits.detach()

    def rsample(self, sample_shape=(), temperature=1.0):
        # (..., K)
        return F.gumbel_softmax(self.logits, tau=temperature, hard=True, dim=-1)

    def sample(self, **kwargs):
        raise NotImplementedError


class MultiOneHotDist:
    def __init__(self, logits, shape, unimix_ratio=0.0):
        self.shape = shape
        splits = torch.split(logits, shape, dim=-1)
        self.onehots = [OneHotDist(s, unimix_ratio=unimix_ratio) for s in splits]

    @property
    def mode(self):
        _modes = [dist.mode for dist in self.onehots]
        return torch.cat(_modes, dim=-1)

    def rsample(self, sample_shape=()):
        _rsamples = [dist.rsample() for dist in self.onehots]
        return torch.cat(_rsamples, dim=-1)

    def sample(self, **kwargs):
        raise NotImplementedError

    def log_prob(self, value):
        splits = torch.split(value, self.shape, dim=-1)
        _log_probs = [dist.log_prob(s) for dist, s in zip(self.onehots, splits)]
        return sum(_log_probs)

    def entropy(self):
        _entropies = [dist.entropy() for dist in self.onehots]
        return sum(_entropies)


class TwoHot:
    def __init__(self, logits, bins, squash=None, unsquash=None):
        # (..., N_bins), (N_bins,)
        self.logits = to_f32(logits)
        assert self.logits.shape[-1] == len(bins), (self.logits.shape, len(bins))

        self.bins = bins
        self.probs = F.softmax(self.logits, dim=-1)  # (..., N_bins)
        self.squash = squash if squash is not None else (lambda x: x)
        self.unsquash = unsquash if unsquash is not None else (lambda x: x)

    def mode(self):
        # (..., N_bins), (N_bins,) -> (..., 1)
        n = self.logits.shape[-1]
        if n % 2 == 1:
            m = (n - 1) // 2
            p1 = self.probs[..., :m]
            p2 = self.probs[..., m : m + 1]
            p3 = self.probs[..., m + 1 :]
            b1 = self.bins[..., :m]
            b2 = self.bins[..., m : m + 1]
            b3 = self.bins[..., m + 1 :]
            wavg = (p2 * b2).sum(dim=-1, keepdim=True) + ((p1 * b1).flip(dims=(-1,)) + (p3 * b3)).sum(
                dim=-1, keepdim=True
            )
            return self.unsquash(wavg)
        p1 = self.probs[..., : n // 2]
        p2 = self.probs[..., n // 2 :]
        b1 = self.bins[..., : n // 2]
        b2 = self.bins[..., n // 2 :]
        wavg = ((p1 * b1).flip(dims=(-1,)) + (p2 * b2)).sum(dim=-1, keepdim=True)
        return self.unsquash(wavg)

    def log_prob(self, target):
        # (..., 1)
        assert target.dtype == self.probs.dtype
        target = target.squeeze(-1)  # (...,)
        target_squashed = self.squash(target).detach()  # (...,)
        # below/above: (...,)
        below = to_i32(self.bins <= target_squashed.unsqueeze(-1)).sum(dim=-1) - 1
        above = len(self.bins) - to_i32(self.bins > target_squashed.unsqueeze(-1)).sum(dim=-1)
        below = torch.clamp(below, 0, len(self.bins) - 1)
        above = torch.clamp(above, 0, len(self.bins) - 1)
        equal = below == above
        dist_to_below = torch.where(
            equal,
            torch.tensor(1.0, device=target.device, dtype=torch.float32),
            (self.bins[below] - target_squashed).abs(),
        )
        dist_to_above = torch.where(
            equal,
            torch.tensor(1.0, device=target.device, dtype=torch.float32),
            (self.bins[above] - target_squashed).abs(),
        )
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        oh_below = to_f32(F.one_hot(below, num_classes=len(self.bins)))
        oh_above = to_f32(F.one_hot(above, num_classes=len(self.bins)))
        # (..., N_bins)
        mixed_target = oh_below * weight_below.unsqueeze(-1) + oh_above * weight_above.unsqueeze(-1)
        log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdim=True)  # (..., N_bins)
        return (mixed_target * log_pred).sum(dim=-1)  # (...)


class MSEDist:
    def __init__(self, mode, agg="sum"):
        # (..., D)
        self._mode = to_f32(mode)
        self._agg = agg

    def mode(self):
        return self._mode

    def mean(self):
        return self._mode

    def log_prob(self, value):
        # (..., D)
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        assert self._mode.dtype == value.dtype, (self._mode.dtype, value.dtype)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss  # (...)


class SymlogDist:
    def __init__(self, mode, dist="mse", agg="sum", tol=1e-8):
        # (..., D)
        self._mode = to_f32(mode)
        self._dist = dist
        self._agg = agg
        self._tol = tol

    def mode(self):
        return symexp(self._mode)

    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value):
        # (..., D)
        assert self._mode.shape == value.shape
        assert self._mode.dtype == value.dtype
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2.0
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._agg == "mean":
            loss = distance.mean(list(range(len(distance.shape)))[2:])
        elif self._agg == "sum":
            loss = distance.sum(list(range(len(distance.shape)))[2:])
        else:
            raise NotImplementedError(self._agg)
        return -loss  # (...)


class Bound:
    def __init__(self, dist):
        super().__init__()
        self._dist = dist

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    @property
    def mode(self):
        out = self._dist.mean
        return out / torch.clip(torch.abs(out), min=1.0).detach()

    def sample(self, sample_shape=()):
        out = self._dist.rsample(sample_shape)
        return out / torch.clip(torch.abs(out), min=1.0).detach()

    def rsample(self, sample_shape=()):
        """Reparameterized sample constrained to the environment action range."""
        out = self._dist.rsample(sample_shape)
        return out / torch.clip(torch.abs(out), min=1.0).detach()

    def log_prob(self, x):
        return self._dist.log_prob(x)


class TanhNormal:
    """Reparameterized Normal followed by a numerically stable tanh transform.

    ``torch.distributions.TransformedDistribution`` provides the same sampling
    semantics, but does not implement entropy for a tanh transform.  The actor
    loss uses entropy directly, so this small wrapper keeps the exact
    change-of-variables density and estimates transformed entropy with a
    reparameterized Monte Carlo sample.
    """

    def __init__(self, mean, std, entropy_samples=1):
        self._mean = to_f32(mean)
        self._std = to_f32(std)
        self._base = torchd.normal.Normal(self._mean, self._std)
        self._entropy_samples = int(entropy_samples)
        if self._entropy_samples < 1:
            raise ValueError("entropy_samples must be at least one")

    @property
    def mode(self):
        # tanh is monotonic, so the transformed mode used for deterministic
        # control is the transform of the raw Normal mean.
        return torch.tanh(self._mean)

    @property
    def batch_shape(self):
        return self._base.batch_shape[:-1]

    @property
    def event_shape(self):
        return self._base.batch_shape[-1:]

    @staticmethod
    def _log_abs_det_jacobian(pre_tanh):
        # log(1 - tanh(x)^2), written without forming tanh(x)^2 so it
        # remains finite for large magnitudes.
        return 2.0 * (torch.log(torch.tensor(2.0, dtype=pre_tanh.dtype, device=pre_tanh.device))
                      - pre_tanh - F.softplus(-2.0 * pre_tanh))

    @staticmethod
    def _squash(pre_tanh):
        action = torch.tanh(pre_tanh)
        # In exact arithmetic tanh has support (-1, 1). Float32 tanh can round
        # to an endpoint, so move only those rounded values one ULP inward.
        one = torch.ones((), dtype=action.dtype, device=action.device)
        limit = torch.nextafter(one, torch.zeros_like(one))
        return torch.clamp(action, min=-limit, max=limit)

    @staticmethod
    def _unsquash(action):
        one = torch.ones((), dtype=action.dtype, device=action.device)
        limit = torch.nextafter(one, torch.zeros_like(one))
        safe_action = torch.clamp(action, min=-limit, max=limit)
        return torch.atanh(safe_action)

    def rsample(self, sample_shape=()):
        pre_tanh = self._base.rsample(torch.Size(sample_shape))
        return self._squash(pre_tanh)

    def sample(self, sample_shape=()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def log_prob(self, action):
        action = to_f32(action)
        pre_tanh = self._unsquash(action)
        elementwise = self._base.log_prob(pre_tanh) - self._log_abs_det_jacobian(pre_tanh)
        result = elementwise.sum(dim=-1)
        # Values outside the closed numerical action contract are impossible.
        outside = (torch.abs(action) > 1.0).any(dim=-1)
        return torch.where(outside, torch.full_like(result, -torch.inf), result)

    def entropy(self):
        sample_shape = (self._entropy_samples,)
        pre_tanh = self._base.rsample(sample_shape)
        elementwise_log_prob = self._base.log_prob(pre_tanh) - self._log_abs_det_jacobian(pre_tanh)
        estimate = -elementwise_log_prob.sum(dim=-1)
        return estimate.mean(dim=0)


def bounded_normal(x, min_std, max_std, **kwargs):
    mean, std = torch.chunk(x, 2, dim=-1)
    std = (max_std - min_std) * torch.sigmoid(std + 2.0) + min_std
    # The raw mean parameterizes the pre-tanh Normal. Applying tanh to the mean
    # before sampling and clipping afterwards is not a squashed Normal and gives
    # an incorrect policy density at the action limits.
    return TanhNormal(to_f32(mean), to_f32(std))


def normal_std_fixed(mean, std, **kwargs):
    dist = torchd.normal.Normal(to_f32(mean), to_f32(std))
    return Bound(torchd.independent.Independent(dist, 1))


def onehot(mean, unimix_ratio, **kwargs):
    return OneHotDist(to_f32(mean), unimix_ratio=unimix_ratio)


def multi_onehot(mean, unimix_ratio, shape, **kwargs):
    return MultiOneHotDist(to_f32(mean), shape, unimix_ratio=unimix_ratio)


def binary(logits, **kwargs):
    return torchd.independent.Independent(torchd.bernoulli.Bernoulli(logits=to_f32(logits)), 1)


def symexp_twohot(logits, bin_num, **kwargs):
    if bin_num % 2 == 1:
        half = torch.linspace(-20, 0, (bin_num - 1) // 2 + 1, dtype=torch.float32, device=logits.device)
        half = symexp(half)
        bins = torch.concatenate([half, -half[:-1].flip(dims=(0,))], 0)
    else:
        half = torch.linspace(-20, 0, bin_num // 2, dtype=torch.float32, device=logits.device)
        half = symexp(half)
        bins = torch.concatenate([half, -half.flip(dims=(0,))], 0)
    return TwoHot(to_f32(logits), bins)


def symlog_mse(logits, **kwargs):
    return SymlogDist(to_f32(logits))


def mse(logits, **kwargs):
    return MSEDist(to_f32(logits))


def identity(logits, **kwargs):
    return logits


def kl(logits_left, logits_right):
    # (..., K), (..., K)
    logprob_left = torch.log_softmax(logits_left, -1)
    logprob_right = torch.log_softmax(logits_right, -1)
    prob = torch.softmax(logits_left, -1)
    return (prob * (logprob_left - logprob_right)).sum(-1)  # (...)
