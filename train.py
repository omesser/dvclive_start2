import time
import math
import random

from dvclive import Live

params = {"learning_rate": 0.002, "optimizer": "Adam", "epochs": 20}

with Live(save_dvc_exp=True) as live:

  # log a parameters
  for param in params:
    live.log_param(param, params[param])

  # simulate training
  offset = random.uniform(0.02, 0.1)
  for epoch in range(1, params["epochs"] + 1):
    fuzz = random.uniform(0.01, 0.1)
    accuracy = 1 - 2 ** - epoch - fuzz - offset
    loss = 2 ** - epoch + fuzz + offset

    # log metrics to studio
    live.log_metric("accuracy", accuracy)
    live.log_metric("loss", loss)
    live.next_step()
    time.sleep(0.2)

# live.log_artifact("model.pkl", type="model")


