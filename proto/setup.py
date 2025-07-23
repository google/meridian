# Copyright 2025 The Meridian Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The setup.py file for MMM Proto Schema."""

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from setuptools import Command, setup
from setuptools.command.build import build

if sys.version_info[:2] >= (3, 11):
  from tomllib import load as toml_load
else:
  # for python<3.11
  from tomli import load as toml_load


class ProtoBuild(Command):
  """Custom command to build proto files."""

  def initialize_options(self):
    with open("pyproject.toml", "rb") as f:
      cfg = toml_load(f).get("tool", {}).get("unified_schema_builder")
      self._root = Path(*cfg.get("proto_root").split("/"))
      self._deps = {
          url.split("/")[-1].split(".")[0]: url
          for url in cfg.get("github_includes")
      }
      self._srcs = list(self._root.rglob("*.proto"))

  def finalize_options(self):
    pass

  def _check_protoc_version(self):
    out = subprocess.run(
        "protoc --version".split(), check=True, text=True, capture_output=True
    ).stdout
    out = out.strip() if out else ""
    if out.startswith("libprotoc"):
      return int(out.split()[1].split(".")[0])
    return 0

  def _run_cmds(self, commands):
    for c in commands:
      cmd_str = " ".join(c)
      print(f"Running command {cmd_str}")
      try:
        subprocess.run(c, capture_output=True, text=True, check=True)
      except subprocess.CalledProcessError as e:
        print(
            "Skipping Unified Schema compilation since command"
            f" {cmd_str} failed:\n{e.stderr.strip()}"
        )
        return e.returncode
    return 0

  def _compile_proto_in_place(self, includes):
    i = [f"-I{include_path}" for include_path in includes]
    srcs_folders = [src for src in self._srcs]
    commands = [
        ["protoc"] + i + f"--python_out=. {src}".split() for src in srcs_folders
    ]
    return self._run_cmds(commands)

  def _pull_deps(self, root):
    cmds = []
    for folder, url in self._deps.items():
      target_path = root / folder
      target_path.mkdir(parents=True, exist_ok=True)
      cmds.append(f"git clone --quiet {url} {target_path}".split())
    return self._run_cmds(cmds)

  def run(self):
    protoc_major_version = self._check_protoc_version()
    if protoc_major_version < 27:
      print(
          "Skipping Unified Schema compilation since the existing compiler"
          f" version is {protoc_major_version}, which is lower than 27"
      )
      return

    with TemporaryDirectory() as t:
      temp_root = Path(t)
      if self._pull_deps(temp_root):
        return
      includes = [self._root] + [temp_root / path for path in self._deps.keys()]
      if self._compile_proto_in_place(includes):
        return


class CustomBuild(build):
  sub_commands = [("compile_proto_schema", None)] + build.sub_commands


if __name__ == "__main__":
  setup(
      cmdclass={
          "build": CustomBuild,
          "compile_proto_schema": ProtoBuild,
      }
  )
