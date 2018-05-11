import shlex
import subprocess


def git_hash():
    """
    Return the current git hash (shortened).
    :return hash:  String,
    """
    cmd = 'git log -n 1 --pretty="%h"'
    hash = subprocess.check_output(shlex.split(cmd)).strip()
    return hash
