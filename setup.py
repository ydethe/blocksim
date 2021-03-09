import os
import codecs
from setuptools import setup
from setuptools.config import read_configuration
import distutils.cmd
import distutils.log
import subprocess
import shutil
import pathlib


on_rtd = os.environ.get("READTHEDOCS") == "True"

conf_pth = os.path.join(os.path.dirname(os.path.abspath(__file__)), "setup.cfg")
conf_dict = read_configuration(conf_pth)

if on_rtd:
    req = []
else:
    with codecs.open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt"),
        mode="r",
        encoding="utf-8",
    ) as f:
        req = f.read().strip().split("\n")


class BuildSphinxCommand(distutils.cmd.Command):
    description = "build sphinx documentation"
    user_options = []

    def initialize_options(self):
        """Set default values for options."""
        # Each user option must be listed here with their default value.
        pass

    def finalize_options(self):
        """Post-process options."""
        pass

    def run_command(self, command):
        p = subprocess.Popen(command)
        p.wait()

    def run(self):
        """Run command."""
        # sphinx-build -T -b html -D language=fr . _build/html
        command = ["sphinx-build"]
        command.append("-T")
        command.append("-b")
        command.append("html")
        command.append("-D")
        command.append("language=fr")
        command.append(".")
        command.append("_build/html")
        self.announce("Running command: %s" % str(command), level=distutils.log.INFO)
        wd = os.getcwd()
        os.chdir("docs")
        os.makedirs("_build", exist_ok=True)
        self.run_command(command)
        src = "_build/html"
        dst = "/home/user-data/www/git.johncloud.fr/"
        if os.path.exists(dst):
            for dirpath, dirnames, filenames in os.walk(src):
                for f in filenames:
                    d2 = os.path.relpath(dirpath, src)
                    f2 = os.path.join(dirpath, f)
                    dir_dst = os.path.join(dst, d2)
                    t2 = os.path.join(dir_dst, f)
                    pathlib.Path(dir_dst).mkdir(parents=True, exist_ok=True)
                    shutil.copy(f2, t2)
        os.chdir(wd)


setup(
    **conf_dict["option"],
    **conf_dict["metadata"],
    install_requires=req,
    cmdclass={
        "doc": BuildSphinxCommand,
    },
)
