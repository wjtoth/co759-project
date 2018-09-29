import os
import re
from time import time
from pprint import pprint

import numpy as np
import tensorflow as tf

from search_train import load_checkpoint, Metrics, perturb_scheduler


timestamp_regex = re.compile(r"\d{10}")
tp_rule_regex = re.compile(r"<TPRule\.(.+?): \d{1,2}>")
object_regex = re.compile(
    r"('(?:grad_tp_rule|modules|activations|loss_functions"
    r"|perturb_scheduler)': )((?:<|\[).*?(?:>|\])(,|\}))", flags=re.DOTALL)
functools_regex = re.compile(r"(functools\.partial\(.*?\))", flags=re.DOTALL)


def clean_hparams_string(hparams_string):
    cleaned_hparams = object_regex.subn(r"\1'\2'\3", hparams_string)[0]
    cleaned_hparams = functools_regex.subn(r"'\1'", cleaned_hparams)[0]
    return cleaned_hparams.replace("\n", "")


def load_tensorboard_data(file_path):
    data = []
    for event in tf.train.summary_iterator(file_path):
        step_data = {}
        for value in event.summary.value:
            step_data[value.tag] = value.simple_value
        data.append((event.step, step_data))
    return sorted(data, key=lambda step_data: step_data[0])


def convert_checkpoint(checkpoint_data):
    training_metrics = checkpoint_data["training_metrics"]
    eval_metrics = checkpoint_data["eval_metrics"]
    args = checkpoint_data["args"]
    train_batches = training_metrics["dataset_batches"]
    data = []
    for i in range(len(training_metrics["accuracy"])):
        step_data = {"train/" + key: value for key, value in training_metrics[i].items()}
        if i % (train_batches//10) == 0:
            eval_step = i // (train_batches//10)
            if eval_metrics is not None and eval_step < len(eval_metrics["accuracy"]):
                step_data.update({"eval/" + key: value 
                                  for key, value in eval_metrics[eval_step].items()})
        data.append((i*10 + 1, step_data))
    return data, args


def load_experiment_data(experiment_dir=None, most_hours_since=0, 
                         min_hours_since=0, most_recent=0, 
                         timestamps=None, hparams=None, root_dir=None):
    if experiment_dir is not None:
        file_names = os.listdir(experiment_dir)
        tb_file_names = [file_name for file_name in file_names 
                         if "tfevents" in file_name]
        tb_file_name = tb_file_names[0] if tb_file_names else None
        checkpoint_file_names = [file_name for file_name in file_names 
                                 if "checkpoint" in file_name]
        if (tb_file_name is None and not checkpoint_file_names 
                or "hparams.txt" not in file_names):
            return 
        if tb_file_name is not None:
            try:
                data = load_tensorboard_data(
                    os.path.join(experiment_dir, tb_file_name))
            except:
                return
        if checkpoint_file_names:
            checkpoint_data = load_checkpoint(experiment_dir)
            data, experiment_hparams = convert_checkpoint(checkpoint_data)
        if "hparams.txt" in file_names:
            hparams_file = open(os.path.join(experiment_dir, "hparams.txt"))
            experiment_hparams = eval(clean_hparams_string(hparams_file.read()).strip())
        return {"hparams": experiment_hparams, "data": data}

    if root_dir is None:
        root_dir = "."
    if (most_hours_since is None and min_hours_since is None 
            and most_recent is None and timestamps is None):
        raise ValueError("No arguments provided.")
    if timestamps is None:
        timestamps = []

    experiment_data = {}
    for dir_path, dir_names, file_names in os.walk(root_dir):
        if "1" in dir_path[-10:] and timestamp_regex.fullmatch(dir_path[-10:]):
            timestamp = int(dir_path[-10:])
            current_time = time()
            if (current_time - (3600*most_hours_since) <= timestamp 
                    <= current_time - (3600*min_hours_since) or timestamp in timestamps
                    or hparams is not None):
                print("Loading experiment:", timestamp)
                data = load_experiment_data(experiment_dir=dir_path)
                if data is None:
                    continue
                checkpoint_match = None
                if hparams is not None and "last_checkpoint" in hparams:
                    checkpoint_match = any([str(hparams["last_checkpoint"]) in file 
                                            for file in file_names])
                if (hparams is not None and all([data["hparams"][hparam] == value 
                        for hparam, value in hparams.items() 
                        if value is not None and hparam != "last_checkpoint"]) 
                        and (checkpoint_match is None or checkpoint_match)):
                    for hparam in hparams:
                        if hparam != "last_checkpoint":
                            data[hparam] = data["hparams"][hparam]
                        else:
                            data[hparam] = hparams[hparam]
                    experiment_data[str(timestamp)] = data
                elif hparams is None:
                    experiment_data[str(timestamp)] = data
        if most_recent != 0 and any(timestamp_regex.fullmatch(name) for name in dir_names):
            experiment_dirs = sorted([name for name in dir_names])
            for directory in experiment_dirs[-most_recent:]:
                data = load_experiment_data(
                    experiment_dir=os.path.join(dir_path, directory))
                experiment_data[directory] = data

    return experiment_data


def get_experiment_stats(experiment_data, include_global_stats=True, 
                         include_hparams=False, pretty_print=True, 
                         to_file=False, append=False):
    stats = {timestamp: {} for timestamp in experiment_data}
    for timestamp, data_dict in experiment_data.items():
        hparams, data = data_dict["hparams"], data_dict["data"]
        eval_steps = list(enumerate(
            [step for step in data if "eval/accuracy" in step[1]]))
        if eval_steps:
            last_epoch, last_step = eval_steps[-1]
            best_epoch, best_step = max(
                eval_steps, key=lambda step: step[1][1]["eval/accuracy"])
            best_step[1]["step"] = (best_step[0], best_epoch)
            last_step[1]["step"] = (last_step[0], last_epoch)
            stats[timestamp]["best"] = best_step[1]
            stats[timestamp]["last"] = last_step[1]
        if include_hparams:
            stats[timestamp]["hparams"] = hparams
        for key, value in data_dict.items():
            if key not in ["hparams", "data"]:
                stats[timestamp][key] = value

    if include_global_stats:
        # Includes mean and standard error of the mean
        global_stats = {}
        for metric in stats[list(stats.keys())[0]]["best"]:
            global_stats[metric] = {}
            values = [data["best"][metric] for data in stats.values()]
            mean = np.mean(values)
            std_error = np.std(values) / np.sqrt(len(values))
            global_stats[metric]["mean"] = mean
            global_stats[metric]["std_error"] = std_error
        stats["global_stats"] = global_stats
 
    if pretty_print:
        pprint(stats, stream=None if not to_file 
               else open(to_file, "a+" if append else "w+"))
        if to_file:
            print("\n", file=open(to_file, "a+"))

    return stats

