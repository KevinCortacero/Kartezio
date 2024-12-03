from kartezio.endpoint import EndpointThreshold
from kartezio.evolution.base import KartezioSequentialTrainer
from kartezio.fitness import FitnessIOU
from kartezio.libraries.array import create_array_lib
from kartezio.utils.dataset import one_cell_dataset


def main():
    """
    Main function to demonstrate building and training a model using Kartezio.

    This example is intended to help new users get started with setting up and training a CGP-based model.
    """
    # Define the number of inputs and create required components
    n_inputs = 1
    libraries = create_array_lib()  # Create a library of array operations
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
    model.set_mutation_rates(node_rate=0.05, out_rate=0.1)

    # Load training data
    train_x, train_y = one_cell_dataset()  # Use a simple one-cell dataset for training

    # Train the model
    elite, history = model.fit(train_x, train_y)

    # Evaluate the model
    evaluation_result = model.evaluate(train_x, train_y)
    print(f"Model Evaluation Result: {evaluation_result}")

    # Export the model as a Python class
    model.print_python_class("MyExampleClass")


if __name__ == "__main__":
    main()
