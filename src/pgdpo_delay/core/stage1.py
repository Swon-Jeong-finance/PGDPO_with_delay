"""Stage I: common LSTM-DPO warm-up (solver layer, torch port pending).
Problems supply simulate/running_cost/terminal_cost; the training loop,
optimiser schedule, and checkpointing live here once, shared by P1-P4."""
def train_stage1(problem, config, seed):      # pragma: no cover - solver pending
    raise NotImplementedError("Stage I torch implementation is the next solver-layer task")
