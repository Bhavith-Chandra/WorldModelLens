import torch

from world_model_lens.backends.base_adapter import AdapterConfig
from world_model_lens.backends.tdmpc2 import SimNorm, TDMPC2Adapter


def make_adapter() -> TDMPC2Adapter:
    config = AdapterConfig(
        d_obs=7,
        d_h=16,
        d_z=16,
        d_action=3,
        tdmpc2_enc_dim=8,
        tdmpc2_mlp_dim=16,
        tdmpc2_simnorm_dim=4,
        tdmpc2_num_bins=11,
        tdmpc2_num_q=3,
    )
    return TDMPC2Adapter(config)


def assert_simplex_groups(z: torch.Tensor, group_size: int) -> None:
    groups = z.reshape(*z.shape[:-1], -1, group_size)
    torch.testing.assert_close(groups.sum(dim=-1), torch.ones_like(groups[..., 0]))
    assert torch.all(groups >= 0)


def test_tdmpc2_encoder_and_dynamics_end_in_simnorm():
    adapter = make_adapter()
    obs = torch.randn(2, 7)
    actions = torch.randn(2, 3)

    z, representation = adapter.encode(obs, torch.empty(0))
    next_z = adapter.next(z, actions)

    assert torch.equal(z, representation)
    assert isinstance(adapter.encoder.net[-1].activation, SimNorm)
    assert isinstance(adapter.dynamics_model[-1].activation, SimNorm)
    assert_simplex_groups(z, 4)
    assert_simplex_groups(next_z, 4)


def test_tdmpc2_has_action_conditioned_distributional_heads():
    adapter = make_adapter()
    z = torch.randn(2, 16)
    actions = torch.randn(2, 3)

    reward_logits = adapter.reward(z, actions)
    q_logits = adapter.q_logits(z, actions)
    all_q_values = adapter.q(z, actions, return_type="all")

    assert adapter.reward_model[0].in_features == 16 + 3
    assert reward_logits.shape == (2, 11)
    assert q_logits.shape == (3, 2, 11)
    assert all_q_values.shape == (3, 2, 1)
    assert all(not parameter.requires_grad for parameter in adapter.target_q_ensemble.parameters())


def test_tdmpc2_policy_is_squashed_gaussian_and_initial_state_is_valid():
    adapter = make_adapter()
    z = torch.randn(2, 16)

    action, mean, log_std = adapter.policy(z, sample=False)
    h0, z0 = adapter.initial_state(batch_size=2)

    assert action.shape == mean.shape == log_std.shape == (2, 3)
    assert torch.all(action.abs() <= 1.0)
    assert torch.all(log_std >= adapter.log_std_min)
    assert torch.all(log_std <= adapter.log_std_max)
    torch.testing.assert_close(h0, z0)
    assert_simplex_groups(h0, 4)
