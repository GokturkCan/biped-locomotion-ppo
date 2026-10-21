"""Check dimensions and initialization semantics before checkpoint use."""


def validate_checkpoint(contract, profile):
    expected = (57, 82, 12) if profile == "v24" else (64, 89, 12)
    if (
        tuple(contract[k] for k in ("actor_size", "critic_size", "action_size"))
        != expected
    ):
        raise ValueError("checkpoint observation/action contract mismatch")
    if contract.get("model_binding_id") is None:
        raise ValueError("private runtime model binding ID required")
