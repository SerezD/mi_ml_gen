import torch
import torch.nn as nn
import numpy as np

from src.generators.utils import z_to_w
from src.generators.sampling import truncated_noise_sample


def mse_dist(points_1: np.ndarray, points_2: np.ndarray):

    return np.linalg.norm(points_1 - points_2, axis=1)


class NonlinearWalk(nn.Module):
    """
    original walker from cop_gen github:
    https://github.com/LiYinqi/COP-Gen/blob/master/cop_gen/train_navigator_bigbigan_optimal.py
    """
    def __init__(self, dim_z, reduction_ratio=1.0):

        super(NonlinearWalk, self).__init__()

        self.walker = nn.Sequential(
            nn.Linear(dim_z, int(dim_z / reduction_ratio)),
            nn.ReLU(inplace=True),
            nn.Linear(int(dim_z / reduction_ratio), dim_z)
        )

        # weight initialization: default

    def forward(self, input, **kwargs):
        """
        slightly changed to be consistent with ours (see below)
        """
        return input, self.walker(input), None


class NonlinearWalker(nn.Module):

    def __init__(self, g_type: str, chunks: str, init_weights: bool = True):
        """
        `Walker` T(z) for perturbation of one or more consecutive latents
        :param g_type: one of bigbigan or stylegan
        :param chunks: decide which latents (chunks) to perturb in format 'start_end'
        """
        super().__init__()

        self.g_type = g_type

        if g_type == 'bigbigan':
            self.z_chunk_dim = 20
            self.z_dim = 120

            self.w_chunk_dim = None  # no W space.
        else:
            assert g_type == 'stylegan'
            self.z_chunk_dim = None  # chunks are on W space.
            self.z_dim = 512

            self.w_chunk_dim = 512

        # find feature length based on n chunks 
        self.start, self.stop = [int(n) for n in chunks.split('_')]
        self.n_chunks = self.stop - self.start

        if g_type == 'bigbigan':
            feats = self.n_chunks * self.z_chunk_dim
        else:
            feats = self.z_dim  # transform on whole z space

        self.walker = nn.Sequential(
            nn.Linear(feats, feats),
            nn.ReLU(inplace=True),
            nn.Linear(feats, feats)
        )

        # init as Identity
        if init_weights:
            for m in self.walker.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.normal_(m.weight, 0., 0.01)
                    torch.nn.init.uniform_(m.bias, -0.001, 0.001)

    def forward(self, z_anchor: np.ndarray | torch.Tensor, **kwargs):
        """
        :param z_anchor: (B, DIM)
        :param kwargs: if using stylegan, pass the mapping_net object to project z points to W space
        :return anchor (z or w), positive (z or w), estimated delta (pertubation distance) on batch
        """

        device = self.walker[0].weight.device
        b, _ = z_anchor.shape

        if isinstance(z_anchor, np.ndarray):
            z_anchor = torch.tensor(z_anchor, device=device)

        if self.g_type == 'bigbigan':

            input_chunks = z_anchor[:, self.start * self.z_chunk_dim: self.stop * self.z_chunk_dim]
            out = self.walker(input_chunks)

            # update anchor
            noise = torch.zeros_like(z_anchor, device=device)
            noise[:, self.start * self.z_chunk_dim: self.stop * self.z_chunk_dim] = out
            positive = z_anchor + noise

            # delta estimation = dist(anchor_chunks, positive_chunks)
            delta = mse_dist(input_chunks.detach().cpu().numpy(), (input_chunks + out).detach().cpu().numpy())

            return z_anchor, positive, delta
        else:
            # stylegan case
            mapping_net = kwargs['mapping_net']

            # z to w (B, CHUNKS, DIM)
            w_anchor = z_to_w(mapping_net, z_anchor, device=device)

            # z_noise
            noise = self.walker(z_anchor) + z_anchor
            w_noise = z_to_w(mapping_net, noise, device=device)

            # positive = replace perturbation on desired chunks
            chunks_mask = torch.zeros_like(w_anchor, device=device, dtype=torch.bool)
            chunks_mask[:, self.start:self.stop] = 1

            w_positive = torch.where(chunks_mask, w_noise, w_anchor)

            # estimate delta for batch on w space
            delta = mse_dist(w_anchor[:, self.start:self.stop].detach().view(b, -1).cpu().numpy(),
                             w_positive[:, self.start:self.stop].detach().view(b, -1).cpu().numpy())

            return w_anchor, w_positive, delta


class RandomWalker(nn.Module):
    def __init__(self, g_type: str, chunks: str, gauss_params: tuple):
        """
        Random walker has no params, just a wrapper. 
        Enables consistency with the NonLinearWalker class
        :param g_type: one of bigbigan or stylegan
        :param chunks: decide which latents (chunks) to perturb in format 'start_end'
        :gauss_params: (mean, std, truncation)
        """

        super().__init__()

        self.g_type = g_type
        self.mean, self.std, self.trunc = gauss_params

        if g_type == 'bigbigan':
            self.z_chunk_dim = 20
            self.z_dim = 120

            self.w_chunk_dim = None
        else:
            assert g_type == 'stylegan'
            self.z_chunk_dim = None  # chunks are on W space.
            self.z_dim = 512

            self.w_chunk_dim = 512

        self.start, self.stop = [int(n) for n in chunks.split('_')]
        self.n_chunks = self.stop - self.start

    def forward(self, z_anchor: torch.Tensor, **kwargs):
        """
        :param z_anchor: (B, DIM)
        :param kwargs: use it to pass "seed" for positive sampling and "mapping_net" if using stylegan
        :return anchor (z or w), positive (z or w), estimated delta on batch
        """

        b, dim_z = z_anchor.shape

        seed = kwargs['seed']

        if self.g_type == 'bigbigan':

            input_chunks = z_anchor[:, self.start * self.z_chunk_dim: self.stop * self.z_chunk_dim]
            out = truncated_noise_sample(b, self.z_chunk_dim * self.n_chunks, self.mean, self.std, self.trunc, seed)

            # update anchor
            noise = np.zeros_like(z_anchor)
            noise[:, self.start * self.z_chunk_dim: self.stop * self.z_chunk_dim] = out
            positive = z_anchor + noise

            delta = mse_dist(input_chunks, (input_chunks + out))

            return z_anchor, positive, delta
        else:

            # stylegan case
            mapping_net = kwargs['mapping_net']
            device = mapping_net.device

            # z to w (B, CHUNKS, DIM)
            w_anchor = z_to_w(mapping_net, z_anchor)

            # z_noise
            noise = truncated_noise_sample(b, dim_z, self.mean, self.std, self.trunc, seed) + z_anchor
            w_noise = z_to_w(mapping_net, noise)

            # positive = replace perturbation on desired chunks
            chunks_mask = torch.zeros_like(w_anchor, device=device, dtype=torch.bool)
            chunks_mask[:, self.start:self.stop] = 1

            w_positive = torch.where(chunks_mask, w_noise, w_anchor)

            # estimate delta for batch on w space
            delta = mse_dist(w_anchor[:, self.start:self.stop].view(b, -1).cpu().numpy(),
                             w_positive[:, self.start:self.stop].view(b, -1).cpu().numpy())

            return w_anchor, w_positive, delta


class SynthWalker(nn.Module):
    
    def __init__(self, g_type: str):
        """
        Synthetic walker with no params, just a wrapper (does nothing)
        Enables consistency with the NonLinearWalker class
        :param g_type: one of bigbigan or stylegan
        """

        super().__init__()

        self.g_type = g_type

    def forward(self, z_anchor: torch.Tensor, **kwargs):
        """
        :param z_anchor: (B, DIM)
        :param kwargs: use it to pass "mapping_net" if using stylegan or None
        :return anchor (z or w), positive = anchor (z or w), None (no delta estimation)
        """

        if self.g_type == 'bigbigan':

            return z_anchor, z_anchor, None
        else:

            # stylegan case
            mapping_net = kwargs['mapping_net']

            # z to w (B, CHUNKS, DIM)
            w_anchor = z_to_w(mapping_net, z_anchor)

            return w_anchor, w_anchor, None


class MultipleNonLinearWalker(nn.Module):
    
    def __init__(self, groups: dict):
        """
        Enable different kind of perturbation for different "chunks"
        Requires to pre-train the standard NonLinearWalker. 
        :param groups: key = chunks as "start_end" | value = ckpt path to walker
        """

        super().__init__()

        # "learned" case, Tz is a path to a checkpoint
        self.all_walkers = [torch.load(ckpt)['nn_walker'] if ckpt is not None else None for ckpt in groups.values()]
        self.all_chunks = groups.keys()

    def forward(self, z_anchor: np.ndarray | torch.Tensor, **kwargs):
        """
        :param z_anchor: (B, DIM)
        :param kwargs: if using stylegan, pass the mapping_net object to project z points to W space
        :return anchor (z or w), positive (z or w), estimated delta on batch
        """

        # retrieve device
        i = 0
        while self.all_walkers[i] is None:
            i = i + 1
        device = self.all_walkers[i].walker[0].weight.device

        b, _ = z_anchor.shape

        if isinstance(z_anchor, np.ndarray):
            z_anchor = torch.tensor(z_anchor, device=device)

        mapping_net = kwargs['mapping_net']

        # z to w (B, CHUNKS, DIM)
        w_anchor = z_to_w(mapping_net, z_anchor, device=device)
        w_positive = w_anchor.clone()

        for group, walker in zip(self.all_chunks, self.all_walkers):

            if walker is not None:
                start, stop = [int(n) for n in group.split('_')]

                # z_noise
                noise = walker.walker(z_anchor) + z_anchor
                w_noise = z_to_w(mapping_net, noise, device=device)

                # positive = replace perturbation on desired chunks
                chunks_mask = torch.zeros_like(w_anchor, device=device, dtype=torch.bool)
                chunks_mask[:, start:stop] = 1

                w_positive = torch.where(chunks_mask, w_noise, w_positive)

        return w_anchor, w_positive, None


class MultipleRandomWalker(nn.Module):

    def __init__(self, gen_type: str, groups: dict):
        """
        Enable different kind of perturbation for different "chunks"
        :param groups: key = chunks as "start_end" | value = tuple perturbation (mu, sigma, trunc)
        """

        super().__init__()

        assert gen_type == 'bigbigan' or gen_type == 'stylegan'

        self.gen_type = gen_type
        self.all_chunks = list(groups.keys())
        self.all_perturbations = list(groups.values())
        self.n_chunks = int(self.all_chunks[-1].split('_')[-1])

    def forward(self, z_anchor: np.ndarray | torch.Tensor, **kwargs):
        """
        :param z_anchor: (B, DIM)
        :param kwargs: seed and mapping_net object to project z points to W space
        :return anchor (z or w), positive (z or w), estimated delta on batch
        """

        b, dim_z = z_anchor.shape

        seed = kwargs['seed']

        if self.gen_type == 'bigbigan':

            chunk_len = dim_z // self.n_chunks

            z_positive = z_anchor.copy()
            rng = np.random.default_rng(seed)

            for group, perturb in zip(self.all_chunks, self.all_perturbations):
                start, stop = [int(n) for n in group.split('_')]

                # z_noise
                if len(perturb) == 3:
                    if perturb[1] == 0.:
                        # no noise:
                        continue
                    else:
                        noise = truncated_noise_sample(b, dim_z, perturb[0], perturb[1], perturb[2], seed)
                else:
                    # do we perturb this chunk/group ?
                    if rng.binomial(1, 0.5, 1)[0] == 1:
                        # yes
                        # draw std from perturb
                        std = rng.uniform(low=perturb[0], high=perturb[1], size=1)[0]
                        noise = truncated_noise_sample(b, dim_z, 0., std, 2., seed)
                    else:
                        # no
                        continue
                z_positive[:, start * chunk_len: stop * chunk_len] += noise[:, start * chunk_len: stop * chunk_len]

            return z_anchor, z_positive, None

        else:
            mapping_net = kwargs['mapping_net']
            device = mapping_net.device

            # z to w (B, CHUNKS, DIM)
            w_anchor = z_to_w(mapping_net, z_anchor, device=device)
            w_positive = w_anchor.clone()

            for group, perturb in zip(self.all_chunks, self.all_perturbations):

                start, stop = [int(n) for n in group.split('_')]

                # z_noise
                noise = truncated_noise_sample(b, dim_z, perturb[0], perturb[1], perturb[2], seed) + z_anchor
                w_noise = z_to_w(mapping_net, noise)

                # positive = replace perturbation on desired chunks
                chunks_mask = torch.zeros_like(w_anchor, device=device, dtype=torch.bool)
                chunks_mask[:, start:stop] = 1

                w_positive = torch.where(chunks_mask, w_noise, w_positive)

            return w_anchor, w_positive, None
