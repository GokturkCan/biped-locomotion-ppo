"""Optional Brax PPO integration. Importing this module never starts training."""

from functools import partial


def network_factory(profile):
    from brax.training.agents.ppo import networks
    from brax.training.acme import running_statistics

    if profile == "v24":
        return partial(
            networks.make_ppo_networks,
            policy_hidden_layer_sizes=(256, 256, 128),
            value_hidden_layer_sizes=(256, 256, 128),
            policy_obs_key="state",
            value_obs_key="privileged_state",
        )
    from .source_initialization import masked_preprocessor

    def factory(
        observation_size,
        action_size,
        preprocess_observations_fn=running_statistics.normalize,
        **kwargs,
    ):
        return networks.make_ppo_networks(
            observation_size,
            action_size,
            preprocess_observations_fn=masked_preprocessor(preprocess_observations_fn),
            policy_hidden_layer_sizes=(256, 256, 128),
            value_hidden_layer_sizes=(256, 256, 128),
            policy_obs_key="state",
            value_obs_key="privileged_state",
            **kwargs,
        )

    return factory


def train(spec, config, seed, output_dir):
    from brax.training.agents.ppo import train as ppo
    from .wrappers import wrap_for_training as wrap
    from biped_locomotion.environments.forward_v24 import ForwardV24
    from biped_locomotion.environments.unified_u13 import UnifiedU13

    environment = (ForwardV24 if config.profile == "v24" else UnifiedU13)(spec)
    factory = network_factory(config.profile)
    extra = {}
    if config.profile == "u13":
        from .initialization import initialize_u13
        from biped_locomotion.rewards.unified_contract import U13_INIT_ACTION_STD

        network = factory(environment.observation_size, 12)
        extra = {
            "restore_params": initialize_u13(
                network, environment.observation_size, U13_INIT_ACTION_STD, seed
            ),
            "restore_value_fn": False,
        }
    return ppo.train(
        environment=environment,
        wrap_env_fn=wrap,
        network_factory=factory,
        seed=seed,
        save_checkpoint_path=str(output_dir),
        **config.train_kwargs(),
        **extra,
    )
