"""Known-answer tests for TEM spatial metrics and controls."""

import math

import pytest
import torch
from experiments.tem_spatial_probe import corr, decoder, eta2, maps, measure

from neurosush.htm.grid_cells import hexagonal_rate
from neurosush.spatial import gridness


def test_rsa_identical_is_one():
    rng = torch.Generator().manual_seed(42)
    pos = torch.randint(64, (500,), generator=rng)
    noise = torch.randn(500, 64, generator=rng) * 0.1
    activity = torch.nn.functional.one_hot(pos, 64).double() + noise
    result = measure([(activity, pos, pos), (activity, pos, pos)], repeats=10, si_repeats=4)
    assert result["rsa"] >= 0.999
    assert result["map_corr"] >= 0.999


def test_rsa_independent_is_null():
    num_nodes = 64
    iu = torch.triu_indices(num_nodes, num_nodes, 1)

    def dm(t):
        return (t[:, None] - t[None]).square().nansum(-1).sqrt()[iu[0], iu[1]]

    worlds = []
    for s in range(12):
        pos = torch.arange(num_nodes).repeat(10)
        activity = torch.randn(len(pos), 32, generator=torch.Generator().manual_seed(s))
        worlds.append((activity, pos, pos))

    result = measure(worlds, repeats=10, si_repeats=4)
    assert abs(result["rsa"]) < 0.03

    templates = [maps(a, p)[0] for a, p, _ in worlds]
    sim_rsas = []
    for i in range(len(templates)):
        for j in range(i + 1, len(templates)):
            r = corr(dm(templates[i]), dm(templates[j]))
            if not math.isnan(r):
                sim_rsas.append(r)
    sim_p95 = torch.tensor(sim_rsas).quantile(0.95).item()
    assert abs(result["rsa_null_p95"] - sim_p95) < 0.03


def test_rsa_sign_flip_invariant():
    rng = torch.Generator().manual_seed(99)
    pos = torch.randint(64, (600,), generator=rng)
    activity_a = torch.randn(600, 32, generator=rng)
    flip_mask = torch.rand(32, generator=rng) > 0.5
    activity_b = activity_a.clone()
    activity_b[:, flip_mask] *= -1
    result = measure(
        [(activity_a, pos, pos), (activity_b, pos, pos)],
        repeats=10,
        si_repeats=4,
    )
    assert result["rsa"] >= 0.999
    assert abs(result["map_corr"]) < 0.15


def test_rsa_translation_equivariant_code():
    size = 8
    num_nodes = size * size
    nodes = torch.arange(num_nodes)
    r = nodes // size
    c = nodes % size

    rng = torch.Generator().manual_seed(77)
    g0 = torch.randn(32, generator=rng).double()
    g0 = g0 / g0.norm()

    def make_world_code(x0_r, x0_c):
        dr = (r - x0_r).double()
        dc = (c - x0_c).double()
        feats = []
        for k in range(1, 17):
            theta = dr * (k * 0.2) + dc * (k * 0.3)
            cos_t = theta.cos()
            sin_t = theta.sin()
            g_pair = g0[2 * k - 2 : 2 * k]
            r0 = cos_t * g_pair[0] - sin_t * g_pair[1]
            r1 = sin_t * g_pair[0] + cos_t * g_pair[1]
            feats.append(torch.stack((r0, r1), dim=-1))
        return torch.cat(feats, dim=-1)

    g_a = make_world_code(0, 0)
    g_b = make_world_code(2, 3)

    pos = torch.randint(num_nodes, (600,), generator=rng)
    act_a = g_a[pos]
    act_b = g_b[pos]

    result = measure([(act_a, pos, pos), (act_b, pos, pos)], repeats=10, si_repeats=4)
    assert result["rsa"] > 0.95
    assert result["map_corr"] < 0.3


def test_rsa_node_permutation_kills():
    rng = torch.Generator().manual_seed(55)
    pos = torch.randint(64, (600,), generator=rng)
    activity_a = torch.randn(600, 32, generator=rng)
    perm = torch.randperm(64, generator=rng)
    pos_b = perm[pos]
    result = measure(
        [(activity_a, pos, pos), (activity_a, pos_b, pos)],
        repeats=10,
        si_repeats=4,
    )
    assert result["rsa"] < result["rsa_null_p95"] + 0.05


def test_rdm_unvisited_nodes_are_nan():
    rng = torch.Generator().manual_seed(88)
    num_nodes = 64
    pos_a = torch.randint(10, num_nodes, (800,), generator=rng)
    pos_b = torch.randint(10, num_nodes, (800,), generator=rng)
    node_code = torch.randn(num_nodes, 16, generator=rng)
    act_a = node_code[pos_a] + torch.randn(800, 16, generator=rng) * 0.5
    act_b = node_code[pos_b] + torch.randn(800, 16, generator=rng) * 0.5

    result = measure([(act_a, pos_a, pos_a), (act_b, pos_b, pos_b)], repeats=10, si_repeats=4)

    m_a, _ = maps(act_a, pos_a, 8)
    m_b, _ = maps(act_b, pos_b, 8)
    visited = torch.isfinite(m_a).all(dim=-1) & torch.isfinite(m_b).all(dim=-1)
    mu_a = m_a[visited]
    mu_b = m_b[visited]

    iu = torch.triu_indices(len(mu_a), len(mu_a), 1)
    dm_a = (mu_a[:, None] - mu_a[None]).square().nansum(-1).sqrt()[iu[0], iu[1]]
    dm_b = (mu_b[:, None] - mu_b[None]).square().nansum(-1).sqrt()[iu[0], iu[1]]
    expected_rsa = corr(dm_a, dm_b)

    assert abs(result["rsa"] - expected_rsa) < 0.02
    assert 0.5 < result["rsa"] < 0.99


def test_decoder_ignores_unvisited_templates():
    rng = torch.Generator().manual_seed(44)
    pos = torch.randint(10, 64, (1200,), generator=rng)
    activity = torch.nn.functional.one_hot(pos, 64).double()
    acc, _null = decoder(activity, pos, rng, repeats=10)
    assert acc >= 0.95


def test_gridness_hex_16():
    axis = torch.arange(16, dtype=torch.float64)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    coords = torch.stack((x, y), -1)
    hex_rate = hexagonal_rate(coords, scale=6, orientation=0.2)
    signed_hex = hex_rate - hex_rate.mean()
    score_hex = gridness(signed_hex, annulus=(3, 7))
    assert score_hex.hexagonal >= 0.3

    rng = torch.Generator().manual_seed(42)
    rand_map = torch.randn(16, 16, generator=rng)
    rand_signed = rand_map - rand_map.mean()
    score_rand = gridness(rand_signed, annulus=(3, 7))
    assert score_rand.hexagonal < 0.1


def test_eta2_spatial_variance():
    rng = torch.Generator().manual_seed(2026)
    num_nodes = 64
    size = 8

    pos = torch.randint(num_nodes, (1000,), generator=rng)
    node_codes = torch.randn(num_nodes, 12, generator=rng)
    node_codes = node_codes - node_codes.mean(dim=0, keepdim=True)
    activity_pure = node_codes[pos]
    e2_pure = eta2(activity_pure, pos, size)
    assert (e2_pure >= 0.999).all()

    activity_noise = torch.randn(1000, 12, generator=rng)
    e2_noise = eta2(activity_noise, pos, size)
    assert (e2_noise >= 0.0).all()
    assert (e2_noise <= 1.0).all()
    assert (e2_noise < 0.2).all()

    a_single = activity_pure[:, 0]
    mean_all = a_single.mean()
    tss = (a_single - mean_all).square().sum()
    bss = torch.zeros(())
    for q in range(num_nodes):
        mask = pos == q
        n_q = mask.sum()
        if n_q > 0:
            mean_q = a_single[mask].mean()
            bss += n_q.double() * (mean_q - mean_all).square()
    hand_eta2 = (bss / tss).item()
    assert e2_pure[0].item() == pytest.approx(hand_eta2, abs=1e-5)
