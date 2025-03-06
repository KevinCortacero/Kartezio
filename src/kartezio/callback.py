from abc import ABC
from datetime import datetime
from enum import Enum
from typing import Dict
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
from kartezio.core.components import dump_component

from kartezio.helpers import Observer
from kartezio.utils.json_handler import json_write


def timestamp(ms=True):
    dt = datetime.now()
    if ms:
        return dt.microsecond
    return dt


def uuid():
    return str(uuid4())


def eventid():
    return f"{timestamp()}-{uuid()}".replace(" ", "-")


class Event:
    def __init__(
        self, iteration: int, name: str, content: Dict, force: bool = False
    ):
        self.iteration = iteration
        self.name = name
        self.content = content
        self.force = force

    class Events(Enum):
        NEW_PARENT = "on_new_parent"
        START_STEP = "on_step_start"
        END_STEP = "on_step_end"
        START_LOOP = "on_loop_start"
        END_LOOP = "on_loop_end"


class Callback(Observer, ABC):
    def __init__(self, frequency=1):
        self.frequency = frequency
        self.decoder = None

    def set_decoder(self, decoder):
        self.decoder = decoder

    def update(self, event: Event):
        if event.iteration % self.frequency != 0 and not event.force:
            return


        if event.name == Event.Events.START_LOOP:
            self.on_evolution_start(event.iteration, event.content)
        elif event.name == Event.Events.START_STEP:
            self.on_generation_start(event.iteration, event.content)
        elif event.name == Event.Events.END_STEP:
            self.on_generation_end(event.iteration, event.content)
        elif event.name == Event.Events.END_LOOP:
            self.on_evolution_end(event.iteration, event.content)
        elif event.name == Event.Events.NEW_PARENT:
            self.on_new_parent(event.iteration, event.content)

    def on_new_parent(self, iteration: int, content):
        pass

    def on_evolution_start(self, iteration: int, event_content):
        pass

    def on_generation_start(self, iteration: int, event_content):
        pass

    def on_generation_end(self, iteration: int, event_content):
        pass

    def on_evolution_end(self, iteration: int, event_content):
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
        verbose = f"[G {n:06}] {fitness:.6f} {time:.6f}s {fps}fps"
        print(verbose)

    def on_evolution_end(self, n, e_content):
        fitness, time, fps = self._compute_metrics(e_content)
        verbose = f"[G {n:06}] {fitness:.6f} {time:.6f}s {fps}fps, loop done."
        print(verbose)


class CallbackSaveScores(Callback):
    def __init__(self, filename, dataset, preprocessing, fitness):
        super().__init__()
        self.filename = filename
        self.data = []
        self.fitness = fitness
        self.train_x = dataset.train_x
        self.train_y = dataset.train_y
        self.test_x = dataset.test_x
        self.test_y = dataset.test_y
        if preprocessing:
            self.train_x = preprocessing.call(self.train_x)
            self.test_x = preprocessing.call(self.test_x)

    def _add_new_line(self, iteration, event_content):
        genotype = event_content.individuals[0].genotype
        p_train = self.decoder.decode(genotype, self.train_x)[0]
        self.fitness.mode = "train"
        f_train = self.fitness.batch(self.train_y, [p_train])
        if self.test_x:
            p_test = self.decoder.decode(genotype, self.test_x)[0]
            self.fitness.mode = "test"
            f_test = self.fitness.batch(self.test_y, [p_test])
            self.fitness.mode = "train"
        else:
            f_test = np.nan
        self.data.append([float(iteration), float(f_train), float(f_test)])

    def on_new_parent(self, iteration, event_content):
        self._add_new_line(iteration, event_content)
        np.save(self.filename, np.array(self.data))

    def on_evolution_end(self, iteration, event_content):
        self._add_new_line(iteration, event_content)
        print(
            f"{self.filename} saved. {len(self.data)} lines, last = {self.data[-1]}"
        )
        data = np.array(self.data)
        plt.figure()
        plt.plot(data[:, 0], data[:, 1])
        plt.plot(data[:, 0], data[:, 2])
        plt.savefig(f"{self.filename}.png")


class CallbackSaveElite(Callback):
    def __init__(self, filename, dataset, preprocessing, fitness):
        super().__init__()
        self.filename = filename

        self.dataset = (dataset.__to_dict__() if type(dataset) != dict else dataset )
        self.decoder = None
        self.preprocessing = dump_component(preprocessing) if preprocessing != None else None
        self.fitness = dump_component(fitness)

    def set_decoder(self, decoder):
        self.decoder = dump_component(decoder)

    def on_new_parent(self, iteration, event_content):
        elite = event_content.individuals[0].genotype
        json_data = {
            "iteration": iteration,
            "dataset": self.dataset,
            "elite":elite.__to_dict__(),
            #     {"genotype":
            #     elite.__to_dict__()
            # # "chromosomes": {
            # #     chromo_key: {k: v.__to_dict__() for k, v in chromo_val.sequence.items()}
            # #     for chromo_key, chromo_val in elite._chromosomes.items()
            # #     }
            # },
            "preprocessing": self.preprocessing,
            "decoder": self.decoder,
            "fitness": self.fitness,
        }
        json_write(self.filename, json_data, indent=None)
