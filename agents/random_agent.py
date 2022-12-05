import numpy as np
from environment.Base2048Env import Base2048Env


class RandomAgent:
  def __init__(self, env: Base2048Env, params: dict):
    self.env = env
    self.params = params

  def get_action(self, *_):
    return np.random.choice(list(self.env.ACTION_STRING.keys()))

  def update(self, *_):
    pass
