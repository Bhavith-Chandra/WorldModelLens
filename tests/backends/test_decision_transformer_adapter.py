import torch

from world_model_lens.backends.base_adapter import AdapterConfig
from world_model_lens.backends.decision_transformer import DecisionTransformerAdapter


def make_adapter() -> DecisionTransformerAdapter:
    config = AdapterConfig(
        d_obs=6,
        d_action=2,
        d_embed=16,
        n_layers=2,
        n_heads=4,
        imagination_horizon=4,
    )
    return DecisionTransformerAdapter(config).eval()


def test_decision_transformer_uses_source_token_heads_and_shapes():
    adapter = make_adapter()
    states = torch.randn(2, 4, 6)
    actions = torch.randn(2, 4, 2)
    returns = torch.randn(2, 4, 1)
    timesteps = torch.arange(4).expand(2, -1)

    state_preds, action_preds, return_preds = adapter.forward_trajectory(
        states, actions, returns, timesteps
    )

    assert state_preds.shape == (2, 4, 6)
    assert action_preds.shape == (2, 4, 2)
    assert return_preds.shape == (2, 4, 1)
    assert torch.all(action_preds.abs() <= 1.0)


def test_decision_transformer_action_prediction_is_causal():
    torch.manual_seed(7)
    adapter = make_adapter()
    states = torch.randn(1, 4, 6)
    actions = torch.randn(1, 4, 2)
    returns = torch.randn(1, 4, 1)
    timesteps = torch.arange(4).unsqueeze(0)

    _, baseline_actions, _ = adapter.forward_trajectory(states, actions, returns, timesteps)

    changed_states = states.clone()
    changed_actions = actions.clone()
    changed_returns = returns.clone()
    changed_states[:, 1:] += 100.0
    changed_actions[:, 1:] -= 100.0
    changed_returns[:, 1:] *= -100.0
    _, changed_predictions, _ = adapter.forward_trajectory(
        changed_states,
        changed_actions,
        changed_returns,
        timesteps,
    )

    # a_0 is read from the s_0 token, which cannot attend to later tokens.
    torch.testing.assert_close(baseline_actions[:, 0], changed_predictions[:, 0])


def test_decision_transformer_one_step_adapter_reads_correct_positions():
    adapter = make_adapter()
    state = torch.randn(3, 6)
    action = torch.randn(3, 2)

    next_state = adapter.transition(state, state, action)
    predicted_action = adapter.actor_forward(state, state)

    assert next_state.shape == (3, 6)
    assert predicted_action.shape == (3, 2)
