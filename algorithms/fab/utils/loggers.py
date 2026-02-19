"""Code builds on https://github.com/lollcat/fab-jax"""

import abc
import os
import pathlib
import pickle
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import pandas as pd

LoggingData = Mapping[str, Any]


class Logger(abc.ABC):
    # copied from Acme: https://github.com/deepmind/acme
    """A logger has a `write` method."""

    @abc.abstractmethod
    def write(self, data: LoggingData) -> None:
        """Writes `data` to destination (file, terminal, database, etc)."""

    @abc.abstractmethod
    def close(self) -> None:
        """Closes the logger, not expecting any further write."""


class ListLogger(Logger):
    """Manually save the data to the class in a dict. Currently only supports scalar history
    inputs."""

    def __init__(
        self,
        save: bool = False,
        save_path: str = "/tmp/logging_hist.pkl",
        save_period: int = 100,
    ):
        self.save = save
        self.save_path = save_path
        if save:
            if not pathlib.Path(self.save_path).parent.exists():
                pathlib.Path(self.save_path).parent.mkdir(exist_ok=True, parents=True)
        self.save_period = save_period  # how often to save the logging history
        self.history: Dict[str, List[Union[np.ndarray, float, int]]] = {}
        self.print_warning: bool = False
        self.iter = 0

    def write(self, data: LoggingData) -> None:
        for key, value in data.items():
            if key in self.history:
                try:
                    value = float(value)
                except:
                    pass
                self.history[key].append(value)
            else:  # add key to history for the first time
                if isinstance(value, np.ndarray):
                    assert np.size(value) == 1
                    value = float(value)
                else:
                    if isinstance(value, float) or isinstance(value, int):
                        pass
                    else:
                        if not self.print_warning:
                            print("non numeric history values being saved")
                            self.print_warning = True
                self.history[key] = [value]

        self.iter += 1
        if self.save and (self.iter + 1) % self.save_period == 0:
            pickle.dump(self.history, open(self.save_path, "wb"))  # overwrite with latest version

    def close(self) -> None:
        if self.save:
            pickle.dump(self.history, open(self.save_path, "wb"))


class WandbLogger(Logger):
    """Wandb logger that uses the unified logging interface."""
    
    def __init__(self, **kwargs: Any):
        import wandb
        self.run = wandb.init(**kwargs, reinit=True)
        self.iter: int = 0

    def write(self, data: LoggingData) -> None:
        self.run.log(data, step=self.iter, commit=False)
        self.iter += 1

    def close(self) -> None:
        self.run.finish()


class CometLogger(Logger):
    """Comet.ml logger that uses the unified logging interface."""
    
    def __init__(self, **kwargs: Any):
        import comet_ml
        self.experiment = comet_ml.Experiment(**kwargs)
        self.iter: int = 0

    def write(self, data: LoggingData) -> None:
        for key, value in data.items():
            if value is not None:
                self.experiment.log_metric(key, value, step=self.iter)
        self.iter += 1

    def close(self) -> None:
        self.experiment.end()


class UnifiedLogger(Logger):
    """Logger that supports both wandb and comet_ml through the unified interface."""
    
    def __init__(self, use_wandb: bool = True, use_comet: bool = False, **kwargs: Any):
        self.loggers = []
        self.iter: int = 0
        
        if use_wandb:
            import wandb
            self.wandb_run = wandb.init(**kwargs.get('wandb_config', {}), reinit=True)
            self.loggers.append('wandb')
        
        if use_comet:
            import comet_ml
            comet_config = kwargs.get('comet_config', {})
            self.comet_experiment = comet_ml.Experiment(**comet_config)
            self.loggers.append('comet')

    def write(self, data: LoggingData) -> None:
        if 'wandb' in self.loggers:
            self.wandb_run.log(data, step=self.iter, commit=False)
        if 'comet' in self.loggers:
            for key, value in data.items():
                if value is not None:
                    self.comet_experiment.log_metric(key, value, step=self.iter)
        self.iter += 1

    def close(self) -> None:
        if 'wandb' in self.loggers:
            self.wandb_run.finish()
        if 'comet' in self.loggers:
            self.comet_experiment.end()
