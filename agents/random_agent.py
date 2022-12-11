import numpy as np
from agents.base_agent import BaseAgent
from environment.Base2048Env import Base2048Env


class RandomAgent(BaseAgent):
  def __init__(self, env: Base2048Env, params: dict):
    super().__init__(env, params)

  def get_action(self, *_):
    return np.random.choice(list(self.env.ACTION_STRING.keys()))

  def update(self, *_):
    pass
