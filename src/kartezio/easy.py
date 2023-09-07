import numpy as np
from kartezio.export import GenomeToPython
from kartezio.inference import SingleModel
from kartezio.utils.io import JsonLoader, JsonSaver


def read_genome(filepath):
    return JsonLoader().read_individual(filepath)


def save_genome(filepath, genome, dataset, parser):
    JsonSaver(dataset, parser).save_individual(filepath, genome)


def load_model(filepath, series=False):
    dataset, elite, parser = read_genome(filepath)
    return SingleModel(elite, parser)


def show_graph(model, inputs=None, outputs=None, only_active=True, jupyter=False):
    from kartezio.utils.viewer import KartezioViewer
    viewer = KartezioViewer(
        model.parser.shape, model.parser.function_bundle, model.parser.endpoint
    )
    return viewer.get_graph(
        model.genome,
        inputs=inputs,
        outputs=outputs,
        only_active=only_active,
        jupyter=jupyter,
    )


def generate_python_class(filepath, class_name):
    _, genome, parser = read_genome(filepath=filepath)
    python_writer = GenomeToPython(parser)
    python_writer.to_python_class(class_name, genome)


def python_class(model, class_name):
    python_writer = GenomeToPython(model._model.parser)
    python_writer.to_python_class(class_name, model._model.genome)


def print_stats(values, fitness, set_name):
    print(f"-- Statistics for {set_name}, using {fitness} fitness:")
    print("Min \t Mean +/- SD \t Max")
    stats_min = np.min(values)
    stats_mean = np.mean(values)
    stats_sd = np.std(values)
    stats_max = np.max(values)
    print(f"{stats_min:0.3f} \t {stats_mean:0.3f}+/-{stats_sd:0.3f} \t {stats_max:0.3f}")


def get_model_size(model):
    return model._model.parser.active_size(model._model.genome)


def node_histogram(model):
    return model._model.parser.node_histogram(model._model.genome)
