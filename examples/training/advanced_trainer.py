from kartezio.callback import CallbackVerbose
from kartezio.endpoint import EndpointThreshold
from kartezio.evolution.base import KartezioSequentialTrainer
from kartezio.fitness import FitnessIOU
from kartezio.libraries.array import create_array_lib
from kartezio.libraries.scalar import library_scalar
from kartezio.mutation.behavioral import AccumulateBehavior
from kartezio.mutation.decay import LinearDecay
from kartezio.mutation.edges import MutationEdgesNormal
from kartezio.mutation.effect import MutationNormal
from kartezio.utils.dataset import one_cell_dataset


def main():
    """
    Main function to demonstrate building and training a model using Kartezio.

    This example is intended to help new users get started with setting up and training a CGP-based model.
    """
    # Define the number of inputs and create required components
    n_inputs = 1
    libraries = [
        create_array_lib(use_scalars=False),
        library_scalar,
    ]  # Create a library of array operations
    endpoint = EndpointThreshold(
        128, mode="tozero"
    )  # Define the endpoint for the model
    fitness = FitnessIOU()  # Define the fitness metric

    # Build the model with specified components
    model = KartezioSequentialTrainer(
        n_inputs=n_inputs,
        n_nodes=n_inputs * 10,
        libraries=libraries,
        endpoint=endpoint,
        fitness=fitness,
    )

    # Set the mutation rates, decay, behavior, mutation effect, and required FPS
    model.set_mutation_rates(node_rate=0.2, out_rate=0.1)
    model.set_decay(LinearDecay(0.2, 0.01))
    model.set_behavior(AccumulateBehavior())
    model.set_mutation_effect(MutationNormal(0.5, 0.05))
    model.set_mutation_edges(MutationEdgesNormal(2))
    model.set_required_fps(60)

    callbacks = []  # Define the callbacks for the model
    callbacks.append(CallbackVerbose(frequency=10))

    # model = builder.compile(n_iterations=50, n_children=4, callbacks=callbacks)
    # model.summary()  # Display the model summary

    # Load training data
    train_x, train_y = one_cell_dataset()  # Use a simple one-cell dataset for training

    # trainer = KartezioTrainer(model)

    # Train the model
    elite, history = model.fit(100, train_x, train_y, callbacks=callbacks)

    # Evaluate the model
    evaluation_result = model.model.evaluate(train_x, train_y)
    print(f"Model Evaluation Result: {evaluation_result}")

    # Export the model as a Python class
    model.model.print_python_class("MyExampleClass")


if __name__ == "__main__":
    main()
