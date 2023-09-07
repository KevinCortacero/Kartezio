from enum import Enum

from numena.io.drive import Directory
from numena.time import eventid

from kartezio.enums import JSON_ELITE
from kartezio.model.components import KartezioCallback
from kartezio.utils.io import JsonSaver


class Event(Enum):
    START_STEP = "on_step_start"
    END_STEP = "on_step_end"
    START_LOOP = "on_loop_start"
    END_LOOP = "on_loop_end"


class CallbackVerbose(KartezioCallback):
    def _callback(self, n, e_name, e_content):
        fitness, time = e_content.get_best_fitness()
        if time == 0:
            fps = "'inf' "
        else:
            fps = int(round(1.0 / time))
        if e_name == Event.END_STEP:
            verbose = f"[G {n:04}] {fitness:.4f} {time:.6f}s {fps}fps"
            print(verbose)
        elif e_name == Event.END_LOOP:
            verbose = f"[G {n:04}] {fitness:.4f} {time:.6f}s {fps}fps, loop done."
            print(verbose)


class CallbackSave(KartezioCallback):
    def __init__(self, workdir, dataset, frequency=1):
        super().__init__(frequency)
        self.workdir = Directory(workdir).next(eventid())
        self.dataset = dataset
        self.json_saver = None

    def set_parser(self, parser):
        super().set_parser(parser)
        self.json_saver = JsonSaver(self.dataset, self.parser)

    def save_population(self, population, n):
        filename = f"G{n}.json"
        filepath = self.workdir / filename
        self.json_saver.save_population(filepath, population)

    def save_elite(self, elite):
        filepath = self.workdir / JSON_ELITE
        self.json_saver.save_individual(filepath, elite)

    def _callback(self, n, e_name, e_content):
        if e_name == Event.END_STEP or e_name == Event.END_LOOP:
            self.save_population(e_content.get_individuals(), n)
            self.save_elite(e_content.individuals[0])
