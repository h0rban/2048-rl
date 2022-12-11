import itertools
import numpy as np
from utils.utils import idxmax, argmax
from agents.base_agent import BaseAgent
from environment.Base2048Env import Base2048Env


# def get_num_empty_cells(state, action):
#   # todo
#   return 0
#
#
# def get_num_mergable_cells(state, action):
#   # todo
#   return 0
#
#
# def get_board_monotonicity(state, action):
#   # todo
#   return 0


class TDAgent(BaseAgent):

  def __init__(self, env: Base2048Env, params: dict):
    super().__init__(env, params)

    self.alpha = params['alpha']
    self.gamma = params['gamma']

    self.action_space = np.array(list(env.ACTION_STRING.keys()))
    self.n_actions = len(self.action_space)

    self.tuples = np.array([

      # horizontal tuples
      [0, 1, 2, 3],
      [4, 5, 6, 7],
      [8, 9, 10, 11],
      [12, 13, 14, 15],

      # vertical tuples
      [0, 4, 8, 12],
      [1, 5, 9, 13],
      [2, 6, 10, 14],
      [3, 7, 11, 15],

      # square tuples
      [0, 1, 4, 5],
      [1, 2, 5, 6],
      [2, 3, 6, 7],
      [4, 5, 8, 9],
      [5, 6, 9, 10],
      [6, 7, 10, 11],
      [8, 9, 12, 13],
      [9, 10, 13, 14],
      [10, 11, 14, 15],
    ])

    self.n_tuples = len(self.tuples)

    self.max_power = 15
    self.tuple_dim = 4

    self.permutations = list(itertools.product(range(0, self.max_power + 1), repeat=self.tuple_dim))
    self.tuple2index = {self.permutations[i]: i for i in range(len(self.permutations))}

    self.v = np.zeros(self.n_tuples * self.max_power ** self.tuple_dim)

  def get_tuple_indices(self, state):

    # reshape board to a 1d array
    state = state.reshape(-1)

    indices = np.zeros(self.n_tuples, dtype=int)
    for idx_tuple in range(self.n_tuples):
      t = self.tuples[idx_tuple]
      subset = tuple(state[t])
      idx_permutation = self.tuple2index[subset]
      indices[idx_tuple] = idx_tuple * self.n_tuples + idx_permutation
    return indices

  def evaluate(self, state):
    return self.get_tuple_state_value(state).sum()

  def get_tuple_state_value(self, state):
    tuple_indices = self.get_tuple_indices(state)
    return self.v[tuple_indices]

  def get_action_values(self, state):
    values = np.zeros(self.n_actions)
    for action in self.action_space:
      _, after_state = self.env.get_after_state(state, action)
      values[action] = self.evaluate(after_state)

    return values

  def get_action(self, state):
    return idxmax(self.get_action_values(state))

  # (s, a, r, s′, s′′)
  def update(self, state, action, reward, after_state, next_state):

    # v(s) <- v(s) + alpha * (r + gamma * v(s') - v(s))
    v_next = argmax(self.get_action_values(next_state))
    indices = self.get_tuple_indices(state)
    self.v[indices] += self.alpha * (reward + self.gamma * v_next - self.v[indices])
