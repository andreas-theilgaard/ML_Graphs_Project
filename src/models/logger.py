import numpy as np
import pandas as pd


class LoggerClass(object):
    def __init__(self, runs=None, metrics=None, seeds=None, log=None):
        self.results = pd.DataFrame()
        self.metrics = metrics
        self.runs = runs
        self.current_run = 0
        self.saved_results = None
        self.seeds = seeds
        self.log = log
        self.saved_values = {}

    def start_run(self):
        self.current_run += 1
        self.log.info(
            f"Run {self.current_run}/{self.runs} using seed {self.seeds[self.current_run-1]}"
        )
        self.X = np.empty((0, len(self.metrics) + 1))

    def add_to_run(self, results: np.ndarray):
        self.X = np.vstack((self.X, np.append(results, self.current_run)))

    def end_run(self):
        self.results = pd.concat([self.results, pd.DataFrame(self.X)])

    def save_value(self, key, value):
        self.saved_values[key] = value

    def save_results(self, save_path):
        self.results.columns = self.metrics + ["Run"]
        self.results.reset_index(drop=True, inplace=True)
        self.results.to_json(save_path)

    def logger_load(self, save_path):
        self.saved_results = pd.read_json(save_path)

    def get_statistics(self, metrics: list, directions: list, out=False):
        assert sum([1 for x in directions if x in ["+", "-"]]) == len(
            directions
        ), "direction should be either ['+','-'] depending on the metric should be maximized or minimized"
        results = self.results.copy()
        if isinstance(self.saved_results, pd.DataFrame):
            results = self.saved_results.copy()

        results_for_metrics = {}
        for metric, direction in zip(metrics, directions):
            if direction == "+":
                best_results = results.groupby("Run")[metric].max()
            elif direction == "-":
                best_results = results.groupby("Run")[metric].min()
            else:
                raise ValueError("Direction should be either '+' or '-'")
            last_results = results.groupby("Run")[metric].tail(1)
            arrow = "↓" if direction == "-" else "↑"
            print("")
            self.log.info(
                f"""\nMetric: {metric} {arrow}:\n   Best Result: {best_results.mean():.4f} ± {best_results.std():.4f}\n   Last Result: {last_results.mean():.4f} ± {last_results.std():.4f}"""
            )
            print("")
            results_for_metrics[metric] = best_results

        if out:
            return results_for_metrics


# if __name__ == "__main__":
#     runs = 10
#     epochs = 40
#     metrics = ['Loss','Train Acc.','Val Acc.','Test Acc']
#     directions = ['-','+','+','+']
#     load = True

#     if load:
#         Logger =LoggerClass()
#         Logger.logger_load('test.json')
#         Logger.get_statistics(metrics=metrics,directions=directions)

#     else:
#         Logger =LoggerClass(runs=runs,metrics=metrics,epochs=epochs)

#         for i in range(runs):
#             Logger.start_run()
#             for j in range(epochs):
#                 out_metric = np.random.uniform(0,1,size=4)
#                 Logger.add_to_run(out_metric)
#             Logger.end_run()
#         Logger.save_results('test.json')
#         Logger.get_statistics(metrics=metrics,directions=directions)
