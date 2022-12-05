import numpy as np
import pandas as pd
from tqdm import tqdm
from agents.random_agent import RandomAgent
from environment.Base2048Env import Base2048Env
from utils.plotting import plot_curves
import matplotlib.pyplot as plt



def train(_env, agent_class, params, episodes):

  # object to keep track of statistics
  stats = []

  # create an agent
  _agent = agent_class(_env, params)

  for _ in tqdm(range(episodes)):

    # variables for statistics
    t = 0

    # reset the environment and get the first action
    state = _env.reset()
    action = _agent.get_action(state)

    done = False
    while not done:
      next_state, reward, done, *_ = _env.step(action)
      next_action = _agent.get_action(next_state)
      _agent.update(state, action, reward, next_state, next_action)
      state, action = next_state, next_action

      # update stats
      t += 1

    # add stats to the list
    stats.append({
      'episode_length': t,
      'final_score': _env.score,
      'max_tile': _env.get_board().max(),
    })

  return _agent, pd.DataFrame(stats)


def run_experiment(_env, agent_class, agent_params, **kwargs):
  # default parameters
  trials = kwargs.get('trials', 5)
  episodes = kwargs.get('episodes', int(1e4))

  _results = []
  for _ in range(trials):
    result = train(_env, agent_class, agent_params, episodes)
    _results.append(result)
  return _results


def process_training_results(_results, precision: int = 4):
  # record trained_agents, episode lengths, final scores, max tiles, max tile distributions
  _agents, episode_lengths, final_scores, max_tiles, max_tile_distribution = [], [], [], [], {}
  for _agent, _result in _results:

    # record episode lengths, final scores, max tiles
    episode_lengths.append(_result.episode_length)
    final_scores.append(_result.final_score)
    max_tiles.append(_result.max_tile)

    # # record max tile distribution
    # counts = _result.max_tile.value_counts().to_dict()
    # for k, v in counts.items():
    #   max_tile_distribution[int(k)] = max_tile_distribution.get(k, 0) + v

    # record agent
    _agents.append(_agent)

  # # normalize the distribution
  # total = sum(max_tile_distribution.values())
  # for k, v in max_tile_distribution.items():
  #   max_tile_distribution[k] = round(v / total, precision)

  return {
    'agents': _agents,
    'episode_lengths': episode_lengths,
    'final_scores': final_scores,
    'max_tiles': max_tiles,
    # 'max_tile_distribution': max_tile_distribution,
  }


agents = [

  # RandomAgent,
  {
    'agent_class': RandomAgent,
    'agent_params': {},
    'experiment_params': {
      'trials': 3,
      'episodes': int(1e1),
    },
  },

]

# run experiments, record training results
for index, agent in enumerate(agents):
  exp_results = run_experiment(Base2048Env(), agent['agent_class'], agent['agent_params'], **agent['experiment_params'])
  agents[index] |= {'training_results': process_training_results(exp_results)}


# plot training results
for agent in agents:
  name = agent['agent_class'].__name__
  training_results = agent['training_results']
  n_trials = agent['experiment_params']['trials']

  # plot training result curves
  for field in ['episode_lengths', 'final_scores', 'max_tiles']:
    plot_curves([np.array(training_results[field])], [name], ['blue'], f'{name} {field} over {n_trials} trials')

  # tile_distribution = agent['training_results']['max_tile_distribution']
  # plt.hist(tile_distribution, )
  # plt.show()

# todo agent evaluation - something like tile distribution

print('done')

