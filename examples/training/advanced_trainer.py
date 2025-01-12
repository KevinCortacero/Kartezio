from kartezio.callback import CallbackVerbose
from kartezio.core.endpoints import EndpointThreshold, ThresholdWatershed
from kartezio.core.fitness import IoU
from kartezio.evolution.base import KartezioTrainer
from kartezio.mutation.behavioral import AccumulateBehavior
from kartezio.mutation.decay import DegreeDecay, LinearDecay
from kartezio.mutation.edges import MutationEdgesNormal
from kartezio.mutation.effect import MutationNormal
from kartezio.primitives.array import create_array_lib
from kartezio.primitives.scalar import library_scalar
from kartezio.utils.dataset import one_cell_dataset


def main():
    """
    Main function to demonstrate building and training a model using Kartezio.

    This example is intended to help new users get started with setting up and training a CGP-based model.
    """
    # Define the number of inputs and create required components
    n_inputs = 1
    libraries = [
        create_array_lib(use_scalars=True),
        library_scalar,
    ]  # Create a library of array operations
    endpoint = EndpointThreshold(128)  # Define the endpoint for the model
    endpoint = ThresholdWatershed(True, 192)
    fitness = IoU()  # Define the fitness metric

    # Build the model with specified components
    model = KartezioTrainer(
        n_inputs=n_inputs,
        n_nodes=30,
        libraries=libraries,
        endpoint=endpoint,
        fitness=fitness,
    )

    model.set_mutation_rates(node_rate=0.5, out_rate=0.2)
    model.set_decay(DegreeDecay(4, 0.5, 0.01))
    model.set_behavior(AccumulateBehavior())
    model.set_mutation_effect(MutationNormal(0.5, 0.005))
    model.set_mutation_edges(MutationEdgesNormal(10))
    model.set_required_fps(60)

    callbacks = []  # Define the callbacks for the model
    callbacks.append(CallbackVerbose(frequency=10))

    # model = builder.compile(n_iterations=50, n_children=4, callbacks=callbacks)
    # model.summary()  # Display the model summary

    # Load training data
    train_x, train_y = (
        one_cell_dataset()
    )  # Use a simple one-cell dataset for training

    # trainer = KartezioTrainer(model)

    # Train the model
    elite, history = model.fit(100, train_x, train_y, callbacks=callbacks)

    # Evaluate the model
    evaluation_result = model.evaluate(train_x, train_y)
    print(f"Model Evaluation Result: {evaluation_result}")

    # Export the model as a Python class
    model.print_python_class("MyExampleClass")


if __name__ == "__main__":
    main()
