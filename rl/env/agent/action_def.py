from enum import Enum
import numpy as np
from .config import *
"""
action define
"""
class TaskType(Enum):
    CLAS = 0
    REG = 1

class MoveDir(Enum):
    MoveNil = 0
    Forward = 1
    Left = 2
    Down = 3
    Right = 4
    UL = 5
    DL = 6
    DR = 7
    UR = 8

class TurnLR():
    Type = TaskType.CLAS
    # Left, Nil, Right
    Name =   ["Left5", "Left4","Left3","Left2","Left1","Left0","Nil","Right0","Right1","Right2","Right3","Right4", "Right5"]
    #Degree = [    -20,     -15,    -10,     -5,     -3,     -1,    0,       1,       3,       5,      10,        15,     20]
    Degree = [    -40,     -20,    -12,     -7,     -3,     -1,    0,       1,       3,       7,      12,        20,     40]

class TurnLR_Degree():
    Type = TaskType.REG
    Name = ["Degree"]

class TurnUD():
    Type = TaskType.CLAS
    # Down, Nil, Up
    Name = ["Down3","Down2","Down1","Down0","Nil","Up0","Up1","Up2","Up3"]
    Degree = [   -7,     -3,     -1,   -0.1,    0,  0.1,    1,    3,    7]

class TurnUD_Degree():
    Type = TaskType.REG
    Name = ["Degree"]

class Fire(Enum):
    Nil = 0
    Fire = 1
    Reammo = 2
   
class State(Enum):
    Nil = 0
    Jump = 1
    Crouch = 2
    Stand = 3
   
class Jet(Enum):
    Nil = 0
    Forward = 1
    Up = 2
   
class MoveType(Enum):
    Norm = 0
    Sprint = 1
    Silence = 2

class UseSkill(Enum):
    Nil = 0
    Use = 1

Move_label_name = ["N", "U", "L", "D", "R", "UL", "DL", "DR", "UR"]
class MoveDirMeta():
    def __init__(self):
        pass

    def reverse(action_label):
        return Move_label_name[action_label]

class TurnLRMeta():
    def __init__(self):
        pass

    def reverse(action_label):
        return TurnLR.Degree[action_label]

class TurnUDMeta():
    def __init__(self):
        pass

    def reverse(action_label):
        return TurnUD.Degree[action_label]

class FireMeta():
    def __init__(self):
        pass

class StateMeta():
    def __init__(self):
        pass

class JetMeta():
    def __init__(self):
        pass

class MoveTypeMeta():
    def __init__(self):
        pass

class UseSkillMeta():
    def __init__(self):
        pass
