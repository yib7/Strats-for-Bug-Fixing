"""Experiment-tracking wiring shared by the three trainer entry points.

One decision lives here: whether ``TrainingArguments.report_to`` gets ``["wandb"]``, and
what has to be true of the environment before it does.
"""

from __future__ import annotations

import os

# W&B's SDK boots its own Sentry client from a hardcoded DSN
# (`wandb/analytics/sentry.py`: `o151352.ingest.sentry.io`) and reports its own crashes
# there. That client is built with sentry-sdk's defaults, which include
# `include_local_variables=True`, so a captured exception carries the stack frames' locals --
# and inside `wandb.init` those locals include the `Settings` object, whose repr contains
# `api_key='<your key>'` verbatim.
#
# Measured, not assumed: with a canary `WANDB_API_KEY` exported, one instrumented
# `run_finetune` produced 2 Sentry events whose payload contained the canary in full, and the
# process resolved `o151352.ingest.sentry.io` four times. So a training run that merely fails
# to reach W&B would hand the user's W&B key to a *different* service. Setting this to "false"
# reproduces 0 events and no DNS lookup.
#
# Only a default: someone debugging the W&B SDK can still export
# `WANDB_ERROR_REPORTING=true` and get the old behaviour. `tests/conftest.py` sets the same
# variable for the suite; this is the user-facing half of that guard.
_ERROR_REPORTING = "WANDB_ERROR_REPORTING"


def wandb_report_to() -> list[str]:
    """``["wandb"]`` when ``WANDB_API_KEY`` is set, else ``[]`` -- and never with crash reporting.

    The presence check is the whole configuration surface: `pop` reads the key's value
    nowhere and passes it to nobody. Enabling the integration is what causes the W&B SDK to
    read it, which is why the Sentry opt-out above is applied here rather than in the CLI --
    it has to hold for a library caller too.

    A *blank* `WANDB_ERROR_REPORTING` counts as unset rather than as an override.
    `.env.example` ships every optional variable with an empty value and documents
    `set -a; . ./.env; set +a`, so sourcing it exports the empty string -- the same trap
    `pop.train.precision._env_str` exists to avoid. W&B happens to read `""` as false today,
    which is the safe direction, but this must not depend on how a third party parses a blank.
    """
    if not os.environ.get("WANDB_API_KEY", "").strip():
        return []
    if not os.environ.get(_ERROR_REPORTING, "").strip():
        os.environ[_ERROR_REPORTING] = "false"
    return ["wandb"]
