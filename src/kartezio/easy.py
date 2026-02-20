import numpy as np

from kartezio.utils.io import JsonLoader


def read_genome(filepath):
    return JsonLoader().read_individual(filepath)


def show_graph(model, inputs=None, outputs=None, only_active=True, jupyter=False):
    from kartezio.utils.viewer import KartezioViewer

    viewer = KartezioViewer(
        model.decoder.infos, model.decoder.library, model.decoder.endpoint
    )
    return viewer.get_graph(
        model.genome,
        inputs=inputs,
        outputs=outputs,
        only_active=only_active,
        jupyter=jupyter,
    )


def print_stats(values, fitness, set_name):
    print(f"-- Statistics for {set_name}, using {fitness} fitness:")
    print("Min \t Mean +/- SD \t Max")
    stats_min = np.min(values)
    stats_mean = np.mean(values)
    stats_sd = np.std(values)
    stats_max = np.max(values)
    print(
        f"{stats_min:0.3f} \t {stats_mean:0.3f}+/-{stats_sd:0.3f} \t {stats_max:0.3f}"
    )


def get_model_size(model):
    return model._model.decoder.active_size(model._model.genome)


def node_histogram(model):
    return model._model.decoder.node_histogram(model._model.genome)
