from kartezio.core.builder import ModelBuilder
from kartezio.core.registry import registry
from kartezio.endpoint import EndpointCounting
from kartezio.fitness import FitnessCount
from kartezio.image.bundle import BUNDLE_OPENCV

ENDPOINT_DEFAULT_COUNTING = EndpointCounting(area_range=None, threshold=4)
BUNDLE_DEFAULT_COUNTING = BUNDLE_OPENCV


def create_counting_model(
    generations,
    _lambda,
    area_range=None,
    threshold=1,
    endpoint=ENDPOINT_DEFAULT_COUNTING,
    bundle=BUNDLE_DEFAULT_COUNTING,
    inputs=3,
    nodes=30,
    outputs=1,
    arity=2,
    parameters=2,
    series_mode=False,
    instance_method="random",
    mutation_method="classic",
    node_mutation_rate=0.15,
    output_mutation_rate=0.2,
    use_goldman=True,
    fitness="count",
    secondary_metric="precision",
    callbacks=None,
    dataset_inputs=None,
):
    if type(secondary_metric) == str:
        secondary_metric = registry.metrics.instantiate(secondary_metric)

    if type(fitness) == str:
        if fitness == "count":
            fitness = FitnessCount(secondary_metric=secondary_metric)
        else:
            fitness = registry.fitness.instantiate(fitness)

    builder = ModelBuilder()
    if threshold != 1 or area_range is not None:
        endpoint = EndpointCounting(area_range=area_range, threshold=threshold)
    builder.create(
        endpoint,
        bundle,
        inputs,
        nodes,
        outputs,
        arity,
        parameters,
        series_mode=series_mode,
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
