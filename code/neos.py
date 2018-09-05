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


def generate_commands(displayed_variable="target", baron_options=None):
    command_string = ""
    if baron_options is not None:
        command_string += "option baron_options '"
        for keyword, value in baron_options.items():
            if value is not None:
                command_string += keyword + "=" + str(value) + " "
        command_string = command_string.rstrip() + "';\n"
    command_string += "solve;\n"
    if displayed_variable:
        command_string += "display {}".format(displayed_variable)
    return command_string + ";"


def generate_job(model_file_path, data_file_path, 
                 command_string, to_file=False, job_file_path=None):
    with open(model_file_path) as model_file:
        model_string = model_file.read()
    with open(data_file_path) as data_file:
        data_string = data_file.read()
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
                username=None, password=None, log_output=False):
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
            offset = 0
            status = ""
            output = ""
            while status != "Done":
                time.sleep(1)
                (msg, offset) = neos.getIntermediateResults(job_number, password, offset)
                msg = msg.data.decode()
                output += msg
                if log_output:
                    sys.stdout.write(msg)
                status = neos.getJobStatus(job_number, password)
            msg = neos.getFinalResults(job_number, password).data.decode()
            output += msg
            if log_output:
                sys.stdout.write(msg)
        return output


def parse_output(job_output_string):
    output = job_output_string.split("BARON")[1]
    loss = float(re.search(r"Objective (\d.\d+)", output)[1])
    if ":=" in output:
        target = re.findall(r"\d\D(\d)\D", output.split(":=")[1])
        target = torch.tensor([int(value) for value in target])
    else:
        target = None
    return {"loss": loss, "target": target}


def run_neos_job(model_file_path, data_file_path, display_variable_data=False, 
                 baron_options=None, log_output=False):
    if display_variable_data:
        commands = generate_commands("target", baron_options)
    else:
        commands = generate_commands(None, baron_options)
    job_string = generate_job(model_file_path, data_file_path, commands)
    output = process_job(job_string, log_output=log_output)
    return parse_output(output)


if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()
    results = run_neos_job(args.model_file, args.data_file, 
                           display_variable_data=True, log_output=False)
    print(results)
    print(time.time() - start_time)