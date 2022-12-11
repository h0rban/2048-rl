import random
import numpy as np
from unittest import TestCase


def idxmax(arr: np.array) -> int:
  """
  Returns the index of the maximum value in the array - breaks ties arbitrarily
  :param arr: array with numbers
  :return: index of the random maximum value in the array
  """
  return np.random.choice(np.flatnonzero(arr == arr.max()))


def argmax(arr: np.array) -> int:
  """
  Returns the index of the maximum value in the array - breaks ties arbitrarily
  :param arr: array with numbers
  :return: index of the random maximum value in the array
  """
  return arr[idxmax(arr)]


class TestUtils(TestCase):

  def testIdxMax(self):
    arr = np.array([1, 2, 0, 5, 4])
    self.assertEqual(idxmax(arr), 3)

    arr = np.array([1, 5, 0, 4, 5])
    self.assertIn(idxmax(arr), [1, 4])

  def testArgMax(self):
    arr = np.array([1, 2, 0, 5, 4])
    self.assertEqual(argmax(arr), 5)

    arr = np.array([1, 5, 0, 4, 5])
    self.assertEqual(argmax(arr), 5)
