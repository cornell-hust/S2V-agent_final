from __future__ import annotations


MESSAGE = (
    "saver_v3.core.verifier has been retired. "
    "The active v5 contract no longer supports counterfactual verification or counterfactual_faithfulness."
)


def run_counterfactual_verifier(*args, **kwargs):
    del args, kwargs
    raise RuntimeError(MESSAGE)
