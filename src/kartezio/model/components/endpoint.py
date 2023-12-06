class Endpoint(Node, component="Endpoint"):
    """
    Last node called to produce final outputs. Called in training loop,
    not submitted to evolution.
    """

    def __init__(self, name: str, fn: Callable, inputs, **kwargs):
        super().__init__(fn, **kwargs)
        self.inputs = inputs
        self._register(self.__class__, "Endpoint", name)

    def to_toml(self):
        return {
            "name": self.name,
        }
