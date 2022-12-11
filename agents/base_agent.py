from environment.Base2048Env import Base2048Env


class BaseAgent:
  def __init__(self, env: Base2048Env, params: dict):
    self.env = env
    self.params = params

  def get_action(self, *_) -> int:
    raise NotImplementedError()

  def update(self, *_):
    raise NotImplementedError()
