#!/usr/bin/env python

# Copyright (c) 2017 NEOS-Server
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Python XML-RPC client for NEOS Server
"""

import argparse
import os
import sys
import re
import time
try:
    import xmlrpc.client as xmlrpclib
except ImportError:
    import xmlrpclib

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="https://neos-server.org:3333", 
                        help="URL to NEOS Server XML-RPC interface")
    parser.add_argument("--username", default=os.environ.get("NEOS_USERNAME", None), 
                        help="username for NEOS Server user account")
    parser.add_argument("--password", default=os.environ.get("NEOS_PASSWORD", None), 
                        help="password for NEOS Server user account")
    parser.add_argument("--model-file", default="", 
                        help="file location of AMPL model file")
    parser.add_argument("--data-file", default="", 
                        help="file location of AMPL data file")
    return parser.parse_args()


def generate_commands(displayed_variable="target", baron_options=None, batched_data=False):
    command_string = ""
    if baron_options is not None:
        command_string += "option baron_options '"
        for keyword, value in baron_options.items():
            if value is not None:
                command_string += keyword + "=" + str(value) + " "
            else:
                command_string += keyword + " "
        command_string = command_string.rstrip() + "';"
    if batched_data:
        command_string += "\nfor {e in ELEMENTS}{\nlet E := e;\nsolve;\n}"
    else:
        command_string += "\nsolve;"
    if displayed_variable:
        command_string += "\ndisplay {}".format(displayed_variable) + ";"
    command_string += "\ndisplay _total_solve_time, _ampl_time;"
    return command_string 


def combine_data_files(data_file_paths=None, data_strings=None, 
                       file_name=None, store=True):
    if data_strings is None:
        data_strings = []
        for file_path in data_file_paths:
            with open(file_path) as data_file:
                data_string = data_file.read()
            data_strings.append(data_string)
        batch_size = len(data_file_paths)
    else:
        batch_size = len(data_strings)

    combined_data = data_strings[0].splitlines()
    combined_data[8] = combined_data[8][:-1]
    for i, data_string in enumerate(data_strings[1:]):
        target_string = data_string.splitlines()[8][:-1]
        combined_data.insert(9+i, str(2+i) + target_string[1:])
    combined_data[9+i] += ";"
    combined_data.insert(2, "param BatchSize:={} ;".format(batch_size))
    combined_data = "\n".join(combined_data)

    if store:
        file_name = "data_batch.txt" if file_name is None else file_name
        combined_file_path = os.path.join(os.path.split(data_file_paths[0])[0], file_name)
        with open(combined_file_path, "w+") as data_file:
            data_file.write(combined_data)
    return combined_data


def generate_job(model_file_path, data_string, command_string, 
                 to_file=False, job_file_path=None):
    with open(model_file_path) as model_file:
        model_string = model_file.read()
    job_string = """<document>
        <category>minco</category>
        <solver>BARON</solver>
        <inputMethod>AMPL</inputMethod>

        <model><![CDATA[
        {}
        ]]></model>

        <data><![CDATA[
        {}
        ]]></data>

        <commands><![CDATA[
        {}
        ]]></commands>

        <comments><![CDATA[
      
        ]]></comments>

        </document>""".format(model_string, data_string, command_string)
    if to_file:
        with open(job_file_path, "w+") as job_file:
            job_file.write(job_string)
    return job_string


def process_job(job_string, server="https://neos-server.org:3333", 
                username=None, password=None, log_output=False, log_only_results=True):
    neos = xmlrpclib.ServerProxy(server)
    alive = neos.ping()
    if alive != "NeosServer is alive\n":
        sys.stderr.write("Could not make connection to NEOS Server\n")
        sys.exit(1)

    if job_string == "queue":
        msg = neos.printQueue()
        if log_output:
            sys.stdout.write(msg)
    else:
        if username and password:
            job_number, password = neos.authenticatedSubmitJob(
                job_string, username, password)
        else:
            job_number, password = neos.submitJob(job_string)
        if log_output:
            sys.stdout.write("Job number = %d\nJob password = %s\n" 
                             % (job_number, password))
        if job_number == 0:
            sys.stderr.write("NEOS Server error: %s\n" % password)
            sys.exit(1)
        else:
            output, offset, status, timer = "", 0, "", 0
            while status != "Done":
                status = neos.getJobStatus(job_number, password)
                time.sleep(1)
                timer += 1
                if timer > 300:
                    print("\nNEOS BARON job exceeded time limit:", timer, "seconds")
                    print("Ending...")
                    return
            if not log_only_results:
                (msg, offset) = neos.getIntermediateResults(job_number, password, offset)
                msg = msg.data.decode()
                output += msg
                if log_output:
                    sys.stdout.write(msg)
            msg = neos.getFinalResults(job_number, password).data.decode()
            output += msg
            if log_output:
                sys.stdout.write(msg)
        return output


def parse_output(job_output_string, batched_data=False):
    if batched_data:
        instances = job_output_string.split("BARON")[1:]
        losses = []
        for instance_output in instances:
            loss = float(re.search(r"Objective (\d.\d+)", instance_output)[1])
            losses.append(loss)
        output = instances[-1]
        if ":=" in output:
            batch_size = len(re.findall(r"\d", output.splitlines()[3]))
            output = output.split(":=")[1]
            target_values = re.findall(r"[ \t\r\f\v](\d)", output)
            targets = []
            for i in range(batch_size):
                target = [int(value) for j, value in enumerate(target_values) 
                          if j % batch_size == i]
                targets.append(torch.tensor(target))
        else:
            targets = None
    else:
        output = job_output_string.split("BARON")[-1]
        loss = float(re.search(r"Objective (\d.\d+)", output)[1])
        if ":=" in output:
            target = re.findall(r"\d\D(\d)\D", output.split(":=")[1])
            target = torch.tensor([int(value) for value in target])
        else:
            target = None
        losses, targets = [loss], [target] if target is not None else None
    total_time, ampl_time = None, None
    if "_total_solve_time" in job_output_string:
        total_time = float(re.search(
            r"_total_solve_time = (\d{1,3}.\d+)", job_output_string)[1])
    if "_ampl_time" in job_output_string:
        ampl_time = float(re.search(
            r"_ampl_time = (\d{1,3}.\d+)", job_output_string)[1])
    return {"losses": losses, "targets": targets, 
            "total_time": total_time, "ampl_time": ampl_time}


def run_neos_job(model_file_path, data_file_paths=None, data_strings=None, 
                 display_variable_data=False, baron_options=None, 
                 log_output=False, batched_data=False):
    if isinstance(data_file_paths, list):
        batched_data = True
        data_string = combine_data_files(data_file_paths=data_file_paths)
    elif data_file_paths is not None:
        data_string = open(data_file_paths).read()
    else:
        data_string = combine_data_files(data_strings=data_strings, store=False)
    if display_variable_data:
        commands = generate_commands("target", baron_options, batched_data)
    else:
        commands = generate_commands(None, baron_options, batched_data)
    job_string = generate_job(model_file_path, data_string, commands)
    output = process_job(job_string, log_output=log_output)
    if output is not None:
        output_data = parse_output(output, batched_data)
    else:
        output_data = None
    return output_data


if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()
    results = run_neos_job(args.model_file, args.data_file, 
                           display_variable_data=True, log_output=True,
                           baron_options=None, batched_data=False)
    print(results)
    print(time.time() - start_time)