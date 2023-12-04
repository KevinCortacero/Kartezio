"""Kartezio sources

"""

# from kartezio.endpoint import register_endpoints
from kartezio.fitness import register_fitness
from kartezio.metric import register_metrics
from kartezio.stacker import register_stackers

register_metrics()
register_fitness()
# register_endpoints()
register_stackers()
