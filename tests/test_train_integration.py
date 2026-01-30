"""Integration tests for training and evaluation with different policies."""

import shutil
import sys
import tempfile
from pathlib import Path

import pytest
import torch

import cogames.policy.starter_agent as starter_agent
import cogames.policy.trainable_policy_template as trainable_policy_template
from cogames.cli.mission import get_mission
from cogames.train import train


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoints."""
    temp_dir = tempfile.mkdtemp(prefix="cogames_test_")
    yield Path(temp_dir)
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_env_config():
    """Get a small test game configuration.

    Uses training_facility (13x13 map, 4 agents) instead of machina_1 (88x88 map, 20 agents)
    for faster test execution, especially under x86 emulation.
    """
    return get_mission("cogsguard_arena.basic")[1]


@pytest.mark.timeout(120)
def test_train_lstm_policy(test_env_config, temp_checkpoint_dir):
    """Test training with LSTMPolicy for enough steps to create a meaningful checkpoint."""
    train(
        env_cfg=test_env_config,
        policy_class_path="mettagrid.policy.lstm.LSTMPolicy",
        device=torch.device("cpu"),
        initial_weights_path=None,
        num_steps=10,
        checkpoints_path=temp_checkpoint_dir,
        seed=42,
        batch_size=64,
        minibatch_size=64,
        vector_num_envs=1,
        vector_batch_size=1,
        vector_num_workers=1,
        checkpoint_interval=10,
    )

    # Check that checkpoints were created
    checkpoints = list(temp_checkpoint_dir.rglob("*.pt"))
    assert len(checkpoints) > 0, f"Should have at least one checkpoint in {temp_checkpoint_dir}"

    # Verify checkpoint can be loaded
    checkpoint = checkpoints[0]
    state_dict = torch.load(checkpoint, map_location="cpu")
    assert isinstance(state_dict, dict), "Checkpoint should be a state dict"


# RandomPolicy is not trainable - it returns None from network()
# so we skip testing it with the train function
@pytest.mark.timeout(180)
def test_train_lstm_and_load_policy_data(test_env_config, temp_checkpoint_dir):
    """Test training LSTM policy, then loading it for evaluation."""
    from mettagrid.policy.lstm import LSTMPolicy

    # Train the policy
    train(
        env_cfg=test_env_config,
        policy_class_path="mettagrid.policy.lstm.LSTMPolicy",
        device=torch.device("cpu"),
        initial_weights_path=None,
        num_steps=10,
        checkpoints_path=temp_checkpoint_dir,
        seed=42,
        batch_size=64,
        minibatch_size=64,
        vector_num_envs=1,
        vector_batch_size=1,
        vector_num_workers=1,
        checkpoint_interval=10,
    )

    # Find the saved checkpoint
    checkpoints = list(temp_checkpoint_dir.rglob("*.pt"))
    assert len(checkpoints) > 0, f"Should have at least one checkpoint in {temp_checkpoint_dir}"

    # Load the checkpoint into a new policy
    from mettagrid.policy.policy_env_interface import PolicyEnvInterface

    policy_env_info = PolicyEnvInterface.from_mg_cfg(test_env_config)
    policy = LSTMPolicy(policy_env_info)
    policy.load_policy_data(str(checkpoints[0]))

    # Verify the policy network was loaded successfully
    import torch.nn as nn

    assert isinstance(policy.network(), nn.Module)
    # Verify the network has parameters (was loaded)
    assert sum(p.numel() for p in policy.network().parameters()) > 0


@pytest.mark.timeout(120)
def test_make_policy_trainable_and_train(temp_checkpoint_dir):
    """Test that the trainable policy template can be trained."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        policy_file = tmpdir / "my_trainable_policy.py"

        # Copy the trainable policy template directly (avoids slow subprocess under emulation)
        template_path = Path(trainable_policy_template.__file__)
        shutil.copy2(template_path, policy_file)
        assert policy_file.exists(), "Policy file was not created"

        # Add tmpdir to sys.path so the policy module can be imported
        sys.path.insert(0, str(tmpdir))
        try:
            # Train using the generated policy for a few steps
            # Use training_facility (13x13 map) for faster execution
            train(
                env_cfg=get_mission("cogsguard_arena.basic")[1],
                policy_class_path=f"{policy_file.stem}.MyTrainablePolicy",
                device=torch.device("cpu"),
                initial_weights_path=None,
                num_steps=10,
                checkpoints_path=temp_checkpoint_dir,
                seed=42,
                batch_size=64,
                minibatch_size=64,
                vector_num_envs=1,
                vector_batch_size=1,
                vector_num_workers=1,
                checkpoint_interval=10,
            )

            # Check that checkpoints were created
            checkpoints = list(temp_checkpoint_dir.rglob("*.pt"))
            assert len(checkpoints) > 0, "Training should produce checkpoints"
        finally:
            sys.path.remove(str(tmpdir))


@pytest.mark.timeout(60)
def test_make_policy_scripted_runs():
    """Test that the scripted policy template can be instantiated and run."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        policy_file = tmpdir / "my_scripted_policy.py"

        # Copy the scripted policy template directly (avoids slow subprocess under emulation)
        template_path = Path(starter_agent.__file__)
        shutil.copy2(template_path, policy_file)
        assert policy_file.exists(), "Policy file was not created"

        # Verify the file contains expected content
        content = policy_file.read_text()
        assert "class StarterPolicy" in content
        assert "class StarterCogPolicyImpl" in content
        assert "def step_with_state" in content
        assert "MultiAgentPolicy" in content

        # Ensure the copied template doesn't collide with the built-in starter policy.
        if 'short_names = ["starter"]' in content:
            policy_file.write_text(content.replace('short_names = ["starter"]', 'short_names = ["starter_template"]'))

        # Verify policy can be instantiated
        sys.path.insert(0, str(tmpdir))
        try:
            from mettagrid.policy.loader import initialize_or_load_policy
            from mettagrid.policy.policy import PolicySpec
            from mettagrid.policy.policy_env_interface import PolicyEnvInterface

            # Use training_facility (13x13 map) for faster execution
            env_cfg = get_mission("cogsguard_arena.basic")[1]
            policy_env_info = PolicyEnvInterface.from_mg_cfg(env_cfg)
            policy_spec = PolicySpec(class_path=f"{policy_file.stem}.StarterPolicy")
            policy = initialize_or_load_policy(policy_env_info, policy_spec)

            # Verify it can create agent policies
            agent_policy = policy.agent_policy(0)
            assert agent_policy is not None
        finally:
            sys.path.remove(str(tmpdir))
