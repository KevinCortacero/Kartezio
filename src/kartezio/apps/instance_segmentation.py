from kartezio.endpoint import EndpointWatershed
from kartezio.image.bundle import BUNDLE_OPENCV
from kartezio.model.builder import ModelBuilder
from kartezio.stacker import MeanKartezioStackerForWatershed

ENDPOINT_DEFAULT_INSTANCE_SEGMENTATION = EndpointWatershed()
BUNDLE_DEFAULT_INSTANCE_SEGMENTATION = BUNDLE_OPENCV
STACKER_DEFAULT_INSTANCE_SEGMENTATION = MeanKartezioStackerForWatershed()


def create_instance_segmentation_model(
    generations,
    _lambda,
    endpoint=ENDPOINT_DEFAULT_INSTANCE_SEGMENTATION,
    bundle=BUNDLE_DEFAULT_INSTANCE_SEGMENTATION,
    inputs=3,
    nodes=30,
    outputs=2,
    arity=2,
    parameters=2,
    series_mode=False,
    series_stacker=STACKER_DEFAULT_INSTANCE_SEGMENTATION,
    instance_method="random",
    mutation_method="classic",
    node_mutation_rate=0.15,
    output_mutation_rate=0.2,
    use_goldman=True,
    fitness="AP",
    callbacks=None,
    dataset_inputs=None,
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
    model = builder.compile(
        generations, _lambda, callbacks=callbacks, dataset_inputs=dataset_inputs
    )
    return model
