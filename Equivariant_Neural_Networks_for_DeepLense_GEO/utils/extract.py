import os
import tarfile


def _extract(tar_url, extract_path="."):
    tar = tarfile.open(tar_url, "r")
    for item in tar:
        tar.extract(item, extract_path)
        if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
            _extract(item.name, "./" + item.name[: item.name.rfind("/")])

def extract(tar_url, extract_path=".", delete_tar=False):
    _extract(tar_url, extract_path)
    if delete_tar:
        os.remove(tar_url)
