
from kartezio.callback import CallbackVerbose
from kartezio.core.sequential import ModelSequential
from kartezio.dataset import read_dataset
from kartezio.endpoint import EndpointWatershed
from kartezio.fitness import FitnessAP, FitnessIOU
from kartezio.preprocessing import SelectChannels
from kartezio.vision.primitives import library_opencv

if __name__ == "__main__":
    path = r"/home/kevin.cortacero/Repositories/KartezioPaper/cell_image_library/dataset"
    n_children = 5
    n_iterations = 200
    preprocessing = SelectChannels([1, 2])
    dataset = read_dataset(
        path,
        # indices=[50, 5, 0, 82, 3, 86, 48, 32, 39, 8, 55, 10, 53, 49, 38]
        indices=range(2),
    )
    draft = ModelSequential(
        2,
        60,
        library_opencv,
        FitnessAP(reduction="mean"),
        EndpointWatershed(backend="opencv"),
    )
    draft.set_mutation_rates(0.05, 0.1)
    # draft.set_decay(LinearDecay((0.1 - 0.01) / n_iterations))
    # draft.set_decay(LinearDecay(0))
    model = draft.compile(n_iterations, n_children, callbacks=[CallbackVerbose()])
    train_x = preprocessing.call(dataset.train_x)
    test_x = preprocessing.call(dataset.test_x)
    elite, history = model.fit(train_x, dataset.train_y)
    print(model.evaluate(train_x, dataset.train_y))
    print(model.evaluate(test_x, dataset.test_y))
    print("...")
    model.decoder.endpoint.backend = "skimage"
    print(model.evaluate(train_x, dataset.train_y))
    print(model.evaluate(test_x, dataset.test_y))
    p, t = model.predict(train_x)
    print(FitnessAP(reduction="min").batch(dataset.train_y, [p]))
    print(FitnessAP(reduction="mean").batch(dataset.train_y, [p]))
    print(FitnessAP(reduction="median").batch(dataset.train_y, [p]))
    model.print_python_class("Test")
    print(FitnessIOU(reduction="max").batch(dataset.train_y, [p]))
    print(FitnessIOU(reduction="median").batch(dataset.train_y, [p]))
    print(FitnessIOU(reduction="mean").batch(dataset.train_y, [p]))

    # max
    # [0.19133812] max
    # [0.14837946] mean

    # [0.13167721] max
    # [0.1079319]  mean

    # [0.12816173] max
    # [0.12293354] mean

    # [0.14948116] max
    # [0.12291637] mean

    # [0.14601417] max
    # [0.13429898] mean

    # 2000
    # [0.13695665]
    # [0.12529463]

    # mean
    # [0.23037846] max
    # [0.17648654] mean

    # [0.15041383] max
    # [0.12196098] mean

    # [0.1432259]  max
    # [0.11932229] mean

    # [0.14047663] max
    # [0.12685938] mean

    # 2000
    # [0.14866726] max
    # [0.11997464] mean
