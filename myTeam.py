
# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveAgent', second = 'DefensiveAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

def DFS(x):
    if x < 0: return -1
    if x >= 10: return 10
    if x <= 4: return x+1
    if x == 1: return -9999
    return x

def extractFeatures(F, T, I, S, FL, mazeD, P):
  F['DFST'] = 2.5*DFS(P)
  if T > 0:
    F['nID'] = 0
    F['IDis'] = max(2, F['IDis'])
  if len(I) < 0 and S != 0:
    F['DFST'] = 0
    F['SAT'] = (F['SAT'] + 2)**2
    F['foodCount'] = len(FL)
    F['OFD'] = min(mazeD)
  if len(I) != 0:
    F['SAT'] = 0
    F['DFST'] = 0
  return F

def EIF(F, I, W, X, Y, isPacman):
  for g in I:
    gP = g.getPosition()
    N = game.Actions.getLegalNeighbors(gP, W)
    if (X, Y) == gP:
      if g.scaredTimer == 0:
        F['SG'] = 0
        F['NG'] = 1
      else:
        F['eatFood'] += 2
        F['EG'] += 1
    elif ((X, Y) in N) and g.scaredTimer > 0:
      F['SG'] += 1
    elif isPacman and g.scaredTimer > 0:
      F['SG'] = 0
      F['NG'] += 1
  return F

def EDF(F, T, D, W, X, Y):
  if T == 0:
    for g in D:
      gP = g.getPosition()
      N = game.Actions.getLegalNeighbors(gP, W)
      if (X, Y) == gP:
        F['EIn'] = 1
      elif (X, Y) in N:
        F['CIn'] += 1
  return F

def EEF(F, E, W, X, Y):
  for g in E:
    if g.getPosition() != None:
      gP = g.getPosition()
      N = game.Actions.getLegalNeighbors(gP, W)
      if (X, Y) in N:
        F['CIn'] += -10
        F['EIn'] = -10
      elif (X, Y) == gP:
        F['EIn'] = -10
  return F

def ECF(F, C, X, Y, isPacman):
  for x, y in C:
    if X == x and Y == y and isPacman:
      F['EC'] = 1.0
  return F

def getTF(FL, i, W):
  TF = []
  for f in FL:
    fx, fy = f
    ai = i - i%2
    c1 = fy > (ai/2) * W.height/3
    c2 = fy < ((ai/2)+1) * W.height/3
    if c1 and c2:
      TF.append(f)
  return TF

def FoodF(F, food, FL, i, W, X, Y, MD):
  if not F['NG']:
      if food[X][Y]:
        F['eatFood'] = 1.0
      if len(FL) > 0:
        TF = getTF(FL, i, W)
        if len(TF) == 0:
          TF = FL
        mazeDist = [MD((X, Y), f) for f in TF]
        if min(mazeDist) is not None:
          WD = W.width * W.height
          F['NBF'] = float(min(mazeDist))/WD
  return F


class OffensiveAgent(CaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def __init__(self, index):
    self.index = index
    self.observationHistory = []

  def getSuccessor(self, gameState, action):
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != util.nearestPoint(pos):
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def chooseAction(self, gameState):
    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    return random.choice(bestActions)

  def evaluate(self, gameState, action):
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    walls = gameState.getWalls()
    x, y = gameState.getAgentState(self.index).getPosition()
    dx, dy = game.Actions.directionToVector(action)
    nextx = int(x + dx)
    nexty = int(y + dy)

    #features['successorScore'] = -len(foodList)

    if action==Directions.STOP: features['stuck'] = 1.0

    enemies = [gameState.getAgentState(a) for a in self.getOpponents(gameState)]
    invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    defenders = [a for a in enemies if a.isPacman and a.getPosition() != None]

    isPacman = successor.getAgentState(self.index).isPacman
    timer = gameState.getAgentState(self.index).scaredTimer
    capsules = gameState.getCapsules()
    food = self.getFood(gameState)

    features = EIF(features, invaders, walls, nextx, nexty, isPacman)
    features = EDF(features, timer, defenders, walls, nextx, nexty)
    features = EEF(features, enemies, walls, nextx, nexty)
    features = ECF(features, capsules, nextx, nexty, isPacman)
    features = FoodF(features, food, foodList, self.index, walls, nextx, nexty, self.getMazeDistance)

    features.divideAll(10.0)
    return features

  def getWeights(self, gameState, action):
    return {'EC': 10.0, 'NG': -20, 'EG': 1.0, 'EIn': 5, 'CIn': 0, 'TD': 1.5, 'NBF': -1, 'SG': 0.1, 'stuck': -5, 'eatFood': 1}

class DefensiveAgent(CaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """
  def __init__(self, index):
    self.index = index
    self.observationHistory = []

  def getSuccessor(self, gameState, action):
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != util.nearestPoint(pos):
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def chooseAction(self, gameState):
    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    return random.choice(bestActions)

  def evaluate(self, gameState, action):
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['nID'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['IDis'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    if myState.isPacman or len(invaders) < 0 and successor.getScore() != 0:
      features['onDefense'] = -1

    timer = successor.getAgentState(self.index).scaredTimer
    score = successor.getScore()
    FL = self.getFood(successor).asList()
    mazeD = [self.getMazeDistance(myPos, food) for food in self.getFood(successor).asList()]
    initPos = gameState.getInitialAgentPosition(self.getTeam(gameState)[0])

    features = extractFeatures(features, timer, invaders, score, FL, mazeD, (myPos[0]-initPos[0]))
    features['SAT'] = self.getMazeDistance(gameState.getAgentPosition(self.getTeam(gameState)[0]), gameState.getAgentPosition(self.getTeam(gameState)[1]))

    return features


  def getWeights(self, gameState, action):
    return {'DFST': 3, 'nID': -40000, 'onDefense': 1, 'SAT': 45, 'IDis': -1800, 'foodCount': -20, 'OFD': 0, 'stop': -400, 'reverse': -250}