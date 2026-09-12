from __future__ import annotations


def signed_uniform(rng, minimum_absolute: float, maximum_absolute: float) -> float:
    minimum = abs(float(minimum_absolute))
    maximum = abs(float(maximum_absolute))
    if maximum < minimum:
        raise ValueError("maximum_absolute must be at least minimum_absolute")
    magnitude = float(rng.uniform(minimum, maximum))
    return magnitude if float(rng.random()) >= 0.5 else -magnitude
