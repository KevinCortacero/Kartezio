from kartezio.components.genotype import Genotype
from kartezio.core.decoder import Decoder
from kartezio.mutation.base import Mutation


class MutationBehavior:
    """
    Base class to represent mutation behavior as a wrapper.

    This class provides a mechanism for applying a mutation to a genotype and allows setting different mutation strategies.
    """

    def __init__(self):
        """
        Initialize a MutationBehavior instance without a mutation function.
        """
        self._mutation: Mutation = None
        self.decoder = None

    def set_decoder(self, decoder: Decoder):
        """
        Set the decoder for the mutation behavior.

        Args:
            decoder (Decoder): The decoder used to parse the genotype into active nodes.
        """
        self.decoder = decoder

    def mutate(self, genotype: Genotype) -> Genotype:
        """
        Apply the currently set mutation to the provided genotype.

        Args:
            genotype (Genotype): The genotype to mutate.

        Returns:
            Genotype: The mutated genotype.
        """
        return self._mutation.mutate(genotype)

    def set_mutation(self, mutation: Mutation) -> None:
        """
        Set the mutation function to be used.

        Args:
            mutation (Mutation): The mutation strategy to set.
        """
        self._mutation = mutation


class AccumulateBehavior(MutationBehavior):
    """
    Mutation behavior that accumulates changes in a genotype until an active node change occurs.

    This behavior uses the decoder to parse the genotype into active nodes and continues applying mutations until a
    change in the active nodes is detected.
    """

    def __init__(self):
        """
        Initialize an AccumulateBehavior instance with a specific decoder.

        Args:
            decoder (Decoder): The decoder used to parse the genotype into active nodes.
        """
        super().__init__()

    def mutate(self, genotype: Genotype) -> Genotype:
        """
        Apply mutations to the genotype until the set of active nodes changes.

        Args:
            genotype (Genotype): The genotype to mutate.

        Returns:
            Genotype: The mutated genotype with changed active nodes.
        """
        changed = False
        active_nodes = self.decoder.parse_to_graphs(genotype)
        while not changed:
            genotype = self._mutation.mutate(genotype)
            new_active_nodes = self.decoder.parse_to_graphs(genotype)
            changed = active_nodes != new_active_nodes
        return genotype
