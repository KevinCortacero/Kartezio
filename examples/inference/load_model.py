from kartezio.inference import KartezioModel
from kartezio.core.endpoints import *
from kartezio.core.fitness import *
from kartezio.primitives.matrix import create_array_lib

model = KartezioModel("elite.json")
print("model loaded")