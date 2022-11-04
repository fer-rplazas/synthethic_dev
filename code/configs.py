configs = dict()

configs["beta_gamma"] = {
    "betas": [0.58, 0.5],
    "gammas": [0.5, 0.6],
    "beta_sharpness": None,
    "phase": None,
    "pac": None,
    "cross_pac": None,
    "phase_shift": None,
    "burst_length": None,
}

configs["beta_sharpness"] = {
    "betas": None,
    "gammas": None,
    "beta_sharpness": [0.1, 0.9],
    "phase": None,
    "pac": None,
    "cross_pac": None,
    "phase_shift": None,
    "burst_length": None,
}

configs["phase"] = {
    "betas": None,
    "gammas": None,
    "beta_sharpness": None,
    "phase": [0.45, 0.55],
    "pac": None,
    "cross_pac": None,
    "phase_shift": None,
    "burst_length": None,
}
