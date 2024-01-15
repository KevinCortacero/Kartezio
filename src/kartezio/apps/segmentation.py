# from kartezio.core.builder import ModelBuilder
from kartezio.core.sequential import ModelSequential
from kartezio.endpoint import EndpointThreshold
from kartezio.fitness import FitnessIOU
from kartezio.stacker import StackerMean, a_mean
from kartezio.vision.primitives import library_opencv


def create_segmentation_model(
    max_iterations,
    n_children,
    endpoint=EndpointThreshold(128, mode="tozero"),
    library=library_opencv,
    n_inputs=3,
    n_nodes=30,
    n_outputs=1,
    arity=2,
    n_parameters=2,
    iterative=False,
    aggregation=STACKER_DEFAULT_SEGMENTATION,
    instance_method="random",
    mutation_method="random",
    node_mutation_rate=0.15,
    output_mutation_rate=0.2,
    use_goldman=True,
    fitness="IOU",
    callbacks=None,
):
    if not iterative:
        draft = ModelSequential(
            n_inputs, n_nodes, library, FitnessIOU(reduction="mean")
        )

    model = draft.compile(max_iterations, n_children, callbacks=callbacks)

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
