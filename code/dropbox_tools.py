import os
import time
import datetime

from dropbox import dropbox


DBX_PATH = ("/College Files/Grad School/Discrete Optimization "
            "and Deep Learning/project/co759-project/code")


def upload(path, dbx_path=None, token=None, dbx=None, overwrite=False):
    if dbx is None:
        dbx = dropbox.Dropbox(token)
    if dbx_path is None:
        dbx_path = DBX_PATH
    mode = (dropbox.files.WriteMode.overwrite if overwrite 
            else dropbox.files.WriteMode.add)
    if os.path.isfile(path):
        while '//' in path:
            path = path.replace("//", "/")
        modified_time = os.path.getmtime(path)
        with open(path, "rb") as file:
            data = file.read()
        if path.startswith("." + os.path.sep):
            path = os.path.sep.join(path.split(os.path.sep)[1:])
        path = os.path.join(dbx_path, path).replace(os.path.sep, "/")
        dbx.files_upload(data, path, mode, 
            client_modified=datetime.datetime(*time.gmtime(modified_time)[:6]))
    else:
        for dir_path, dir_names, file_names in os.walk(path):
            for file_name in file_names:
                upload(os.path.join(dir_path, file_name), dbx_path=dbx_path, 
                       token=token, dbx=dbx, overwrite=overwrite)
