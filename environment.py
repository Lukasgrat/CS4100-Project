import random


class GameModel():
  def __init__(self, filename, removeFileName, players = 6):
    with open(filename, "r") as file:
      output: str = file.read()
      self.words = output.split('\n')
    
    self.answer = random.choice(self.words)

  def makeHint(self, guess):
    if(guess in self.words):
      self.hints.append(guess)
      return True
    return False
  def outputHints(self):
    dupeList : list = []
    validHints : list = []
    for hint in self.hints:
      if hint in validHints:
        dupeList.append(hint)
        validHints.remove(hint)
      elif hint not in dupeList:
        validHints.append(hint)
    return validHints
  def guess(self, guessString : str):
    return guessString == self.answer 

class GameModelEnv():
  def __init__(self):
    self.model = GameModel("data/words.txt", "data/noise_words.txt", 3)
    self.reward = 0
  def reset(self):
    self.model = GameModel("data/words.txt", "data/noise_words.txt", 3)
    return self.model.answer
  def start_guessing(self, clues: set[str]):
    self.observation = clues
    self.action_space = self.model.words.copy() * 2
    for observation in self.observation:
      self.action_space.remove(observation)
    return self.observation

  def step(self, action):
    if(self.model.guess(action)):
      return 50
    else:
      return -10

  def actionMap(self):
    return self.model.words

def humanController(playerCount : int = 6):
  model : GameModel = GameModel("data/words.txt", "data/noise_words.txt", playerCount)
  print("The word is: ", model.answer)
  for x in range(1, playerCount):
    while(True):
      if(model.makeHint(input("Player " + str(x + 1) + " send a hint.\n"))):
        break
  print("The hints are ", end=" ")
  for hintStr in model.outputHints():
    print(hintStr, end=", ")
  print(".")
  guessString = input("Player 1, make a guess:")
  if(model.guess(guessString)):
    print("Correct!")
  else:
    print("Incorrect, the actual answer was: ", model.answer)