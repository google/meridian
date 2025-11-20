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

"""The setup.py file for Meridian."""

from pathlib import Path
import sass
from setuptools import Command, setup
from setuptools.command.build import build


class ScssCompileCommand(Command):
  _templates_path = 'meridian/templates'
  _scss_output_mapping = {
      'style.scss': 'style.css',
  }

  def initialize_options(self):
    self.build_lib = None

  def finalize_options(self):
    self.set_undefined_options('build_py', ('build_lib', 'build_lib'))

  def run(self):
    if not self.build_lib:
      return

    for scss, css in self._scss_output_mapping.items():
      self._compile_scss(scss, css)

  def _compile_scss(self, source_filename, dest_filename):
    dirpath = Path(self.build_lib).joinpath(self._templates_path)
    dirpath.mkdir(parents=True, exist_ok=True)
    dest_filepath = dirpath.joinpath(dest_filename)

    source_relpath = self._templates_path + '/' + source_filename

    print(f'compiling {source_relpath} -> {dest_filepath}')

    with open(Path.cwd().joinpath(source_relpath), 'r') as f:
      css = sass.compile(string=f.read())

    dest_filepath.write_text(css, encoding='utf-8')

  def get_outputs(self):
    dirpath = Path(self.build_lib).joinpath(self._templates_path)
    return [
        dirpath.joinpath(css) for _, css in self._scss_output_mapping.items()
    ]


class CustomBuild(build):
  sub_commands = [('compile_scss', None)] + build.sub_commands


if __name__ == '__main__':
  setup(cmdclass={'build': CustomBuild, 'compile_scss': ScssCompileCommand})
