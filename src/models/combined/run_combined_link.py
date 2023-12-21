from src.models.combined.combined_link.combined_link1 import fit_combined1_link

# from src.models.combined.combined_link.combined_link2 import fit_combined2_link
# from src.models.combined.combined_link.combined_link3 import fit_combined3_link
from src.models.combined.combined_link.combined_link4 import fit_combined4_link
from src.models.combined.combined_link.test import fit_combined2_link
from src.models.combined.combined_link.test2 import fit_combined3_link


def fit_combined_link(config, dataset, training_args, Logger, log, seeds, save_path):
    if config.dataset.combined.type == "comb1":
        fit_combined1_link(
            config=config,
            dataset=dataset,
            training_args=training_args,
            Logger=Logger,
            log=log,
            seeds=seeds,
            save_path=save_path,
        )
    elif config.dataset.combined.type == "comb2":
        fit_combined2_link(
            config=config,
            dataset=dataset,
            training_args=training_args,
            Logger=Logger,
            log=log,
            seeds=seeds,
            save_path=save_path,
        )
    elif config.dataset.combined.type == "comb3":
        fit_combined3_link(
            config=config,
            dataset=dataset,
            training_args=training_args,
            Logger=Logger,
            log=log,
            seeds=seeds,
            save_path=save_path,
        )
    elif config.dataset.combined.type == "comb4":
        fit_combined4_link(
            config=config,
            dataset=dataset,
            training_args=training_args,
            Logger=Logger,
            log=log,
            seeds=seeds,
            save_path=save_path,
        )
