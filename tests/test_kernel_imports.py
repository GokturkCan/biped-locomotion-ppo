import pytest


def test_all_extracted_methods_resolve():
    pytest.importorskip("jax")
    pytest.importorskip("mujoco")
    pytest.importorskip("ml_collections")
    from biped_locomotion.environments.base import dispatch, source_function

    for key in dispatch()["registry"]:
        assert callable(source_function(key))
