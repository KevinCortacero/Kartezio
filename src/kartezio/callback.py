from abc import ABC
from datetime import datetime
from enum import Enum
from uuid import uuid4

import numpy as np

from kartezio.core.helpers import Observer
from kartezio.drive.directory import Directory
from kartezio.enums import JSON_ELITE
from kartezio.utils.io import JsonSaver


def timestamp(ms=True):
    dt = datetime.now()
    if ms:
        return dt.microsecond
    return dt


def uuid():
    return str(uuid4())


def eventid():
    return f"{timestamp()}-{uuid()}".replace(" ", "-")


class Event(Enum):
    NEW_PARENT = "on_new_parent"
    START_STEP = "on_step_start"
    END_STEP = "on_step_end"
    START_LOOP = "on_loop_start"
    END_LOOP = "on_loop_end"


class Callback(Observer, ABC):
    def __init__(self, frequency=1):
        self.frequency = frequency
        self.decoder = None

    def set_decoder(self, parser):
        self.decoder = parser

    def update(self, event):
        if event["name"] == Event.START_LOOP:
            self.on_evolution_start(event["n"], event["content"])
        elif event["name"] == Event.START_STEP:
            self.on_generation_start(event["n"], event["content"])
        elif event["name"] == Event.END_STEP:
            self.on_generation_end(event["n"], event["content"])
        elif event["name"] == Event.END_LOOP:
            self.on_evolution_end(event["n"], event["content"])
        elif event["name"] == Event.NEW_PARENT:
            self.on_new_parent(event["n"], event["content"])

        """
        if event["n"] % self.frequency == 0 or event["force"]:
            self._notify(event["n"], event["name"], event["content"])
        """

    def on_new_parent(self, iteration, event_content):
        pass

    def on_evolution_start(self, iteration, event_content):
        pass

    def on_generation_start(self, iteration, event_content):
        pass

    def on_generation_end(self, iteration, event_content):
        pass

    def on_evolution_end(self, iteration, event_content):
        pass


class CallbackVerbose(Callback):
    def _compute_metrics(self, e_content):
        _, fitness, time = e_content.get_best_fitness()
        if time == 0:
            fps = "'inf' "
        else:
            fps = int(round(1.0 / time))
        return fitness, time, fps

    def on_generation_end(self, n, e_content):
        fitness, time, fps = self._compute_metrics(e_content)
        verbose = f"[G {n:04}] {fitness:.4f} {time:.6f}s {fps}fps"
        print(verbose)

    def on_evolution_end(self, n, e_content):
        fitness, time, fps = self._compute_metrics(e_content)
        verbose = f"[G {n:04}] {fitness:.4f} {time:.6f}s {fps}fps, loop done."
        print(verbose)


class CallbackSave(Callback):
    def __init__(self, workdir, dataset, frequency=1):
        super().__init__(frequency)
        self.workdir = Directory(workdir).next(eventid())
        self.dataset = dataset
        self.json_saver = None

    def set_decoder(self, parser):
        super().set_decoder(parser)
        self.json_saver = JsonSaver(self.dataset, self.decoder)

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


class CallbackSaveFitness(Callback):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.data = []

    def on_generation_end(self, n, e_content):
        fitness = e_content.individuals[0].fitness
        self.data.append(fitness)

    def on_evolution_end(self, n, e_content):
        np.save(self.filename, np.array(self.data))
        print(
            f"{self.filename} saved. {len(self.data)} lines, last = {self.data[-1]}"
        )
