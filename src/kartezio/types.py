from typing import List, NewType

Score = NewType("Score", float)
ScoreList = List[Score]

KType = NewType("KType", str)
TypeScalar = KType("scalar")
TypeVector = KType("vector")
TypeArray = KType("array")
TypeLabels = KType("labels")
TypeFourier = KType("fourier")
Type3C = KType("3C")
