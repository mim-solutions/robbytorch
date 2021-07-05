import mlflow
import torch
from livelossplot import PlotLosses
from typing import Optional, List

from . import utils



class Writer(object):

    def log_metrics(self, logs, epoch, epochs, model):
        """
        """
        raise NotImplementedError

    def filter_logs(self, logs):
        return {k: v for k, v in logs.items() if not k.startswith("_")}


class MLFlowWriter(Writer):
    
    def __init__(self, run_name, params, log_per: int = 5):
        self.run_name = run_name
        self.params = utils.flatten_dict(params)
        self.log_per = log_per
    
    def log_metrics(self, logs, epoch, epochs, model):
        logs = self.filter_logs(logs)
        if epoch % self.log_per == 0 or epoch == epochs:
            with mlflow.start_run(run_name=self.run_name):
                mlflow.log_param("epoch", epoch)
                mlflow.log_params(self.params)
                mlflow.log_metrics(logs)


class LiveLossWriter(Writer):
    """Prints some nice charts during traning 
    """

    def __init__(self):
        self.liveloss = PlotLosses()
        
    def log_metrics(self, logs, epoch, epochs, model):
        logs = self.filter_logs(logs)
        self.liveloss.update(logs)
        self.liveloss.send()


class ModelWriter(Writer):
    """Saves model during training
    """

    def __init__(self, save_path: str = "ipython/trained_models/temp", log_per: int = 20):
        self.log_per = log_per
        self.save_path = save_path
        utils.mkdir_and_preserve_group(save_path)
        
    def log_metrics(self, logs, epoch, epochs, model):
        if epoch % self.log_per == 0 or epoch == epochs:
            torch.save(model.state_dict(), f"{self.save_path}/model_epoch_{epoch}.pt")