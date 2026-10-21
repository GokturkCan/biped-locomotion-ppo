"""V18 event-synchronized phase. No new timer or alternate FSM."""


def event_phase(environment, info, legacy):
    from .phase_kernels import ResidualForwardWalkingV18__v18_event_phase

    return ResidualForwardWalkingV18__v18_event_phase(environment, info, legacy)
