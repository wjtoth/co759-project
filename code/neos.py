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
import time
try:
    import xmlrpc.client as xmlrpclib
except ImportError:
    import xmlrpclib


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", help="specify XML file name or queue for action")
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
    parser.add_argument("--command-file", default="", 
                        help="file location of AMPL command file")
    return parser.parse_args()


def generate_job(job_file_path, model_file_path, data_file_path, command_file_path):
    with open(model_file_path) as model_file:
        model_string = model_file.read()
    with open(data_file_path) as data_file:
        data_string = data_file.read()
    if command_file_path:
        with open(command_file_path) as command_file:
            command_string = command_file.read()
    else:
        command_string = ""
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
    with open(job_file_path, "w+") as job_file:
        job_file.write(job_string)


def process_job(action, server="https://neos-server.org:3333", 
                username=None, password=None):
    neos = xmlrpclib.ServerProxy(server)
    alive = neos.ping()
    if alive != "NeosServer is alive\n":
        sys.stderr.write("Could not make connection to NEOS Server\n")
        sys.exit(1)

    if action == "queue":
        msg = neos.printQueue()
        sys.stdout.write(msg)
    else:
        xml = ""
        try:
            xmlfile = open(action, "r")
            buffer = 1
            while buffer:
                buffer = xmlfile.read()
                xml += buffer
            xmlfile.close()
        except IOError as e:
            sys.stderr.write("I/O error(%d): %s\n" % (e.errno, e.strerror))
            sys.exit(1)
        if username and password:
            jobNumber, password = neos.authenticatedSubmitJob(
                xml, username, password)
        else:
            jobNumber, password = neos.submitJob(xml)
        sys.stdout.write("Job number = %d\nJob password = %s\n" % (jobNumber, password))
        if jobNumber == 0:
            sys.stderr.write("NEOS Server error: %s\n" % password)
            sys.exit(1)
        else:
            offset = 0
            status = ""
            while status != "Done":
                time.sleep(1)
                (msg, offset) = neos.getIntermediateResults(jobNumber, password, offset)
                sys.stdout.write(msg.data.decode())
                status = neos.getJobStatus(jobNumber, password)
            msg = neos.getFinalResults(jobNumber, password)
            sys.stdout.write(msg.data.decode())


if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()
    generate_job(args.action, args.model_file, args.data_file, args.command_file)
    process_job(args.action, args.server, args.username, args.password)
    print(time.time() - start_time)