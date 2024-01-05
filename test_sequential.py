import cv2
import numpy as np

from kartezio.callback import CallbackVerbose
from kartezio.core.mutation.decay import LinearDecay, FactorDecay
from kartezio.dataset import read_dataset
from kartezio.endpoint import EndpointWatershed
from kartezio.fitness import FitnessIOU, FitnessAP
from kartezio.preprocessing import SelectChannels
from kartezio.vision.primitives import library_opencv
from kartezio.core.sequential import ModelSequential


if __name__ == "__main__":
    path = r"dataset\1-cell_image_library\dataset"
    n_iterations = 2000
    dataset = read_dataset(path, indices=range(2))
    # dataset = read_dataset(path)
    library_opencv.display()
    draft = ModelSequential(
        2,
        30,
        library_opencv,
        FitnessAP(reduction="mean"),
        endpoint=EndpointWatershed(backend="opencv"),
    )
    draft.set_mutation_rates(0.15, 0.20)
    draft.set_decay(LinearDecay((0.15 - 0.05) / n_iterations))
    # draft.set_decay(FactorDecay(0.99))
    model = draft.compile(n_iterations, 4, callbacks=[CallbackVerbose()])
    train_x = SelectChannels([1, 2]).call(dataset.train_x)
    elite, history = model.fit(train_x, dataset.train_y)
    p, t = model.predict(train_x)
    for i, pi in enumerate(p):
        print(i)
        cv2.imwrite(
            f"labels_{i}.png",
            cv2.applyColorMap(pi[0].astype(np.uint8), cv2.COLORMAP_VIRIDIS),
        )
    print(model.evaluate(train_x, dataset.train_y))
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
