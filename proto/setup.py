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

"""The setup.py file for MMM Unified Schema."""

import logging
import pathlib
import subprocess
import sys
import tempfile

import setuptools
from setuptools.command import build


def _toml_load(path):
  if sys.version_info[:2] >= (3, 11):
    import tomllib  # pylint: disable=g-import-not-at-top

    return tomllib.load(path)
  else:
    # for python<3.11
    import tomli  # pylint: disable=g-import-not-at-top

    return tomli.load(path)


class ProtoBuild(setuptools.Command):
  """Custom command to build proto files."""

  def initialize_options(self):
    with open("pyproject.toml", "rb") as f:
      cfg = _toml_load(f).get("tool", {}).get("unified_schema_builder")
      self._root = pathlib.Path(*cfg.get("proto_root").split("/"))
      self._deps = {}
      for url_with_tag in cfg.get("github_includes"):
        tag = None
        url = url_with_tag
        if "@" in url_with_tag:
          url, tag = url_with_tag.split("@")
        folder = url.split("/")[-1].split(".")[0]
        self._deps[folder] = (url, tag)
      self._srcs = list(self._root.rglob("*.proto"))

  def finalize_options(self):
    pass

  def _check_protoc_version(self):
    out = subprocess.run(
        [sys.executable, "-m", "grpc_tools.protoc", "--version"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    out = out.strip() if out else ""
    if out.startswith("libprotoc"):
      return int(out.split()[1].split(".")[0])
    return 0

  def _run_cmds(self, commands):
    for c in commands:
      cmd_str = " ".join(c)
      logging.info("Running command %s", cmd_str)
      try:
        subprocess.run(c, capture_output=True, text=True, check=True)
      except subprocess.CalledProcessError as e:
        logging.error(
            "Skipping Unified Schema compilation since command '%s'"
            " failed:\n%s",
            cmd_str,
            e.stderr.strip(),
        )
        return e.returncode
    return 0

  def _compile_proto_in_place(self, includes):
    i = [f"-I{include_path}" for include_path in includes]
    srcs_folders = [src for src in self._srcs]
    commands = [
        [sys.executable, "-m", "grpc_tools.protoc"]
        + i
        + f"--python_out=. {src}".split()
        for src in srcs_folders
    ]
    return self._run_cmds(commands)

  def _pull_deps(self, root):
    cmds = []
    for folder, (url, tag) in self._deps.items():
      target_path = root / folder
      target_path.mkdir(parents=True, exist_ok=True)
      if tag:
        cmds.append(
            f"git clone --quiet --depth=1 --branch {tag} {url} {target_path}"
            .split()
        )
      else:
        cmds.append(f"git clone --quiet --depth=1 {url} {target_path}".split())
    return self._run_cmds(cmds)

  def run(self):
    protoc_major_version = self._check_protoc_version()
    if protoc_major_version < 27:
      logging.error(
          "Skipping Unified Schema compilation since the existing compiler"
          " version is %s, which is lower than 27",
          protoc_major_version,
      )
      return

    with tempfile.TemporaryDirectory() as t:
      temp_root = pathlib.Path(t)
      if self._pull_deps(temp_root):
        return
      includes = [self._root] + [temp_root / path for path in self._deps.keys()]
      if self._compile_proto_in_place(includes):
        return


class CustomBuild(build.build):
  sub_commands = [
      ("compile_unified_schema_proto", None)
  ] + build.build.sub_commands


if __name__ == "__main__":
  setuptools.setup(
      cmdclass={
          "build": CustomBuild,
          "compile_unified_schema_proto": ProtoBuild,
      }
  )
