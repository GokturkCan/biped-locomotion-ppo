"""U13 retains physical acquisition rewards; command gate is not success-conditioned."""


def command_requested(speed, yaw_rate, speed_threshold=0.015, yaw_threshold=0.020):
    return float(speed >= speed_threshold or abs(yaw_rate) >= yaw_threshold)
