from kartezio.core.builder import ModelBuilder
from kartezio.fitness import FitnessCrossEntropy
from kartezio.improc.primitives import library_opencv

BUNDLE_DEFAULT_CLASSIFICATION = library_opencv
FITNESS_DEFAULT_CLASSIFICATION = FitnessCrossEntropy()


def create_classification_model(
    generations,
    _lambda,
    labels,
    threshold=1,
    reduce_method="count",
    bundle=BUNDLE_DEFAULT_CLASSIFICATION,
    inputs=3,
    nodes=30,
    arity=2,
    parameters=2,
    series_mode=False,
    endpoint=None,
    instance_method="random",
    mutation_method="classic",
    node_mutation_rate=0.15,
    output_mutation_rate=0.2,
    use_goldman=True,
    fitness=FITNESS_DEFAULT_CLASSIFICATION,
    callbacks=None,
    dataset_inputs=None,
):
    builder = ModelBuilder()
    endpoint = endpoint
    outputs = len(labels)
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
