from kartezio.callback import CallbackSaveElite, CallbackSaveScores, CallbackVerbose
from kartezio.core.base import ModelBuilder
from kartezio.data.dataset import read_dataset


def load_dataset(path, level=None, preprocessing=None):
    filename = f"dataset_{level}.csv" if level is not None else "dataset.csv"
    dataset = read_dataset(path, preview=True, indices=None, filename=filename)
    if preprocessing is not None:
        dataset.train_set.x = preprocessing.call(dataset.train_x)
        dataset.test_set.x = preprocessing.call(dataset.test_x)
    return dataset


def main():
    n_inputs = 3
    libraries = create_array_lib()
    endpoint = EndpointThreshold(128, mode="tozero", normalize=True)
    fitness = FitnessIOU()
    builder = ModelBuilder(
        n_inputs,
        n_inputs * 10,
        libraries,
        endpoint,
        fitness,
    )
    # builder.set_mutation_rates(0.05, 0.1)


if __name__ == "__main__":
    main()
