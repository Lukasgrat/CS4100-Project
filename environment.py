import random
class GameModel():
  def __init__(self, filename, players = 6):
    with open(filename, "r") as file:
      output: str = file.read()
      self.words = output.split('\n')
    self.answer = None
  def startGame(self):
    print("Length of words: ", len(self.words))
    self.answer = random.choice(self.words)
    self.hints = []
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
def humanController(playerCount : int = 6):
  model : GameModel = GameModel("words.txt", playerCount)
  model.startGame()
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
humanController(7)