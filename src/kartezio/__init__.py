"""Kartezio sources

"""

from kartezio.endpoint import register_endpoints
from kartezio.fitness import register_fitness
from kartezio.image.nodes import register_nodes
from kartezio.metric import register_metrics
from kartezio.stacker import register_stackers

register_nodes()
register_metrics()
register_fitness()
register_endpoints()
register_stackers()
