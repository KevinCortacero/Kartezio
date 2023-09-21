from typing import List, NewType

Score = NewType("Score", float)
ScoreList = List[Score]

KType = NewType("KType", str)
TypeScalar = KType("Scalar")
TypeVector = KType("Vector")
TypeArray = KType("Array")
