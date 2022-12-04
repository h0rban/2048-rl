import pygame
import matplotlib.pyplot as plt
from environment.Base2048Env import Base2048Env


def render_plot(_env, _screen):
  fig, _ = _env.render()
  fig.canvas.draw()
  surface = pygame.image.frombuffer(fig.canvas.tostring_rgb(), fig.canvas.get_width_height(), 'RGB')
  _screen.blit(surface, (4, 4))
  pygame.display.update()

  # clears plots
  fig.clear()
  plt.close()
  plt.cla()
  plt.clf()


def try_step(_event, _env):
  def set_and_return(action):
    _env.step(action)
    return True

  if _event.key == pygame.K_LEFT:
    return set_and_return(Base2048Env.LEFT)

  if _event.key == pygame.K_UP:
    return set_and_return(Base2048Env.UP)

  if _event.key == pygame.K_RIGHT:
    return set_and_return(Base2048Env.RIGHT)

  if _event.key == pygame.K_DOWN:
    return set_and_return(Base2048Env.DOWN)


pygame.init()
screen = pygame.display.set_mode((400, 400))
pygame.display.set_caption('2048')
env = Base2048Env(max_invalid_moves=4)
render_plot(env, screen)

done = False
while not done:
  stepped = False
  for event in pygame.event.get():
    if event.type == pygame.KEYDOWN:
      stepped = try_step(event, env)
    if event.type == pygame.QUIT:
      done = True

  if stepped:
    render_plot(env, screen)
    done = env.is_done()

print('Final Score: {}'.format(env.score))
