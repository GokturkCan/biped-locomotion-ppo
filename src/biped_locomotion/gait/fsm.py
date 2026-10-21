"""Public access to the retained expected-side/transfer FSM kernel."""


def expected_side_components(environment, data, info, phase_id, terms, motion):
    from biped_locomotion.environments.base import source_function

    return source_function("ResidualForwardWalkingV9._v9_components")(
        environment, data, info, phase_id, terms, motion
    )


def qualified_air_contact(environment, data, info, phase_id, terms, motion):
    from biped_locomotion.environments.base import source_function

    return source_function("ResidualForwardWalkingV8._v8_components")(
        environment, data, info, phase_id, terms, motion
    )
