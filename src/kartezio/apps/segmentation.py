from kartezio.endpoint import EndpointThreshold
from kartezio.image.bundle import BUNDLE_OPENCV
from kartezio.model.builder import ModelBuilder
from kartezio.stacker import StackerMean

ENDPOINT_DEFAULT_SEGMENTATION = EndpointThreshold(threshold=4)
BUNDLE_DEFAULT_SEGMENTATION = BUNDLE_OPENCV
STACKER_DEFAULT_SEGMENTATION = StackerMean()


def create_segmentation_model(
    generations,
    _lambda,
    endpoint=ENDPOINT_DEFAULT_SEGMENTATION,
    bundle=BUNDLE_DEFAULT_SEGMENTATION,
    inputs=3,
    nodes=30,
    outputs=1,
    arity=2,
    parameters=2,
    series_mode=False,
    series_stacker=STACKER_DEFAULT_SEGMENTATION,
    instance_method="random",
    mutation_method="classic",
    node_mutation_rate=0.15,
    output_mutation_rate=0.2,
    use_goldman=True,
    fitness="IOU",
    callbacks=None,
):
    builder = ModelBuilder()
    builder.create(
        endpoint,
        bundle,
        inputs,
        nodes,
        outputs,
        arity,
        parameters,
        series_mode=series_mode,
        series_stacker=series_stacker,
    )
    builder.set_instance_method(instance_method)
    builder.set_mutation_method(
        mutation_method,
        node_mutation_rate,
        output_mutation_rate,
        use_goldman=use_goldman,
    )
    builder.set_fitness(fitness)
    model = builder.compile(generations, _lambda, callbacks=callbacks)
    return model
