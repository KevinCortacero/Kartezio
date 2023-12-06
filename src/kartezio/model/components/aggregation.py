class Aggregation(Node, ABC, component="Aggregation"):
    def __init__(
        self,
        aggregation: Callable,
        inputs,
        post_aggregation: Callable = None,
    ):
        super().__init__(aggregation)
        self.post_aggregation = post_aggregation
        self.inputs = inputs

    def call(self, x: List):
        y = []
        for i in range(len(self.inputs)):
            if self.post_aggregation:
                y.append(self.post_aggregation(self._fn(x[i])))
            else:
                y.append(self._fn(x[i]))
        return y
