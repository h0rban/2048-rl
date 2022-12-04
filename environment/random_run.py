from environment.Base2048Env import Base2048Env

env = Base2048Env()
env.seed(42)

env.reset()
env.render(True)

moves, done = 0, False

while not done:
  action = env.np_random.choice(range(4), 1).item()
  next_state, reward, done, *_ = env.step(action)
  moves += 1

  print(f'Next Action: "{Base2048Env.ACTION_STRING[action]}"\n\nReward: {reward}')
  print(env.get_board())

  print('\nTotal Moves: {}'.format(moves))

env.render(True)
