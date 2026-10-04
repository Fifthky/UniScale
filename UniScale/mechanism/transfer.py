"""Fixed-site independent-history transfer and recovery with matched controls."""

from __future__ import annotations

import numpy as np

from .data import context, generator, queries

OBJECTIVES = ("transfer", "rescue")


def matched_noise(rng, reference):
    noise = rng.standard_normal(reference.shape).astype(np.float32)
    axes = tuple(range(1, noise.ndim))
    return noise * (np.linalg.norm(reference.reshape(len(reference), -1), axis=1)
                    / np.maximum(np.linalg.norm(noise.reshape(len(noise), -1), axis=1), 1e-12))[
                        (slice(None),) + (None,) * len(axes)]


class InterventionSession:
    """Reuse the same donor and recipient states for both registered objectives."""

    def __init__(self, config, model, process, magnitude, query, repeat,
                 forward_fn=None, patch_size=32):
        from .activation import forward

        self.forward, self.model = forward if forward_fn is None else forward_fn, model
        self.settings = config["activation_transfer"]
        self.config = {**config, "data_seed": self.settings["query_seed"]}
        self.process, self.magnitude, self.query, self.repeat = process, magnitude, query, repeat
        self.layer, self.length = self.settings["candidate"]["layer"], self.settings["candidate"]["length"]
        if 96 % patch_size or self.length % patch_size:
            raise ValueError("Query tokens must align exactly between short and long inputs")
        self.local_tokens = list(range(96 // patch_size))
        self.full_tokens = list(range((self.length - 96) // patch_size, self.length // patch_size))
        self.local, self.local_state = self.capture(query, self.local_tokens)

    def capture(self, values, tokens):
        prediction, states = self.forward(self.model, values, tokens, [self.layer])
        return prediction, states[self.layer]

    def patch(self, values, tokens, state):
        return self.forward(self.model, values, tokens,
                            intervention={"layer": self.layer, "values": state})[0]

    def shuffle(self, config, values, seed):
        order = generator(config, self.process, self.magnitude, seed,
                          1, self.repeat, self.length).permutation(self.length - 96)
        shuffled = values.copy()
        shuffled[:, :self.length - 96] = values[:, order]
        return shuffled

    def recipient(self, sign):
        values = context(self.config, self.process, self.magnitude, self.query,
                         self.length, sign, self.repeat, 8001)
        natural, natural_state = self.capture(values, self.full_tokens)
        return {"input": values, "natural": natural, "natural_state": natural_state,
                "null_state": self.local_state,
                "erased": self.patch(values, self.full_tokens, self.local_state)}

    def donor_pool(self, sign):
        donor_config = {**self.config, "data_seed": self.settings["donor_seed"]}
        running = {}
        count = self.settings["candidate"]["donor_count"]
        for member in range(count):
            code = 100 + 10 * self.repeat + member
            carrier, _ = queries(donor_config, self.process, self.magnitude,
                                 10000 + code, len(self.query) // 2)
            _, short_state = self.capture(carrier, self.local_tokens)
            observed = context(donor_config, self.process, self.magnitude, carrier,
                               self.length, sign, 0, 30000 + code)
            _, observed_state = self.capture(observed, self.full_tokens)
            _, shuffled_state = self.capture(self.shuffle(donor_config, observed, 40000 + member),
                                              self.full_tokens)
            alternate, _ = queries(donor_config, self.process, self.magnitude,
                                   20000 + code, len(self.query) // 2)
            _, alternate_state = self.capture(alternate, self.local_tokens)
            changes = {"delta": observed_state - short_state,
                       "carrier_only": short_state - alternate_state,
                       "shuffled_history": shuffled_state - short_state}
            for name, value in changes.items():
                if not np.isfinite(value).all():
                    raise FloatingPointError("Nonfinite donor state difference")
                running[name] = value.copy() if name not in running else running[name] + value
        return {name: value / count for name, value in running.items()}

    def predict(self, objective, recipient, delta):
        change = self.settings["candidate"]["strength"] * delta
        if objective == "transfer":
            return self.patch(self.query, self.local_tokens, self.local_state + change)
        if objective != "rescue":
            raise ValueError("Unknown intervention objective")
        return self.patch(recipient["input"], self.full_tokens, recipient["null_state"] + change)

    def controls(self, recipient, pool, objective, prediction):
        count = self.settings["candidate"]["donor_count"]
        noise = matched_noise(generator(self.config, self.process, self.magnitude, 60000,
                              1, self.repeat, count), pool["delta"])
        result = {"local": self.local, "natural": recipient["natural"], "erased": recipient["erased"],
                  "intervention": prediction,
                  "random_transfer": self.predict(objective, recipient, noise),
                  "carrier_only": self.predict(objective, recipient, pool["carrier_only"]),
                  "shuffled_history": self.predict(objective, recipient, pool["shuffled_history"])}
        restored = self.patch(recipient["input"], self.full_tokens, recipient["natural_state"])
        zero_local = self.patch(self.query, self.local_tokens, self.local_state)
        if not np.allclose(restored, recipient["natural"], atol=1e-5, rtol=1e-5):
            raise RuntimeError("Natural-state restoration failed")
        if not np.allclose(zero_local, self.local, atol=1e-5, rtol=1e-5):
            raise RuntimeError("Short-state restoration failed")
        perturbation = matched_noise(generator(self.config, self.process, self.magnitude, 61000,
                                    1, self.repeat), recipient["null_state"] - recipient["natural_state"])
        result.update(restored=restored, zero_transfer=zero_local,
                      matched_random_erasure=self.patch(recipient["input"], self.full_tokens,
                                                        recipient["natural_state"] + perturbation))
        if any(not np.isfinite(value).all() for value in result.values()):
            raise FloatingPointError("Nonfinite intervention prediction")
        return result
