from src.models.combined.combined_class.combined_class1 import fit_combined1_class
from src.models.combined.combined_class.combined_class2 import fit_combined2_class


def fit_combined_class(config, dataset, training_args, Logger, log, seeds, save_path):
    if config.dataset.combined.type == "comb1":
        fit_combined1_class(
            config=config,
            dataset=dataset,
            training_args=training_args,
            Logger=Logger,
            log=log,
            seeds=seeds,
            save_path=save_path,
        )
    elif config.dataset.combined.type == "comb2":
        fit_combined2_class(
            config=config,
            dataset=dataset,
            training_args=training_args,
            Logger=Logger,
            log=log,
            seeds=seeds,
            save_path=save_path,
        )
