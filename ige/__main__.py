from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gin
from absl import app
from absl import flags
# from absl import logging
from ige import runners

flags.DEFINE_string(
    'command', default='train',
    help='function to run. Should be configured under the "main" module. '
         'Those in `ige.runners` ("train", "evaluate", "vis") imported '
         'automatically')
flags.DEFINE_string(
  'gin_file', None, 'List of paths to the config files.')
flags.DEFINE_multi_string(
  'gin_param', None, 'Newline separated list of Gin parameter bindings.')
flags.DEFINE_string('config_dir', None,
                    'root directory of .gin configs. We change to this '
                    'directory for loading .gin files, so gin files that '
                    'import other gin files should import relative to this '
                    'directory. If None, checks $IGE_CONFIG_DIR and if still'
                    'None uses IGE_DIR/config, where IGE_DIR is the root '
                    'directory in the git repository. If installed via pip '
                    'without -e flag, this may cause issues.')

FLAGS = flags.FLAGS

class ChangeDirContext(object):
    def __init__(self, folder):
        self._target_folder = folder
        self._original_folder = None
    
    def __enter__(self):
        self._original_folder = os.getcwd()
        os.chdir(self._target_folder)
    
    def __exit__(self, *args, **kwargs):
        os.chdir(self._original_folder)
        self._original_folder = None


def main(argv):
    config_dir = FLAGS.config_dir
    if config_dir is None:
        config_dir = os.environ.get('IGE_CONFIG_DIR', None)
    if config_dir is None:
        config_dir = os.path.realpath(
            os.path.join(os.path.dirname(__file__), '..', 'config'))

    with ChangeDirContext(config_dir):
        gin_file = FLAGS.gin_file
        if gin_file is None:
            gin_files = []
        else:
            if not gin_file.endswith('.gin'):
                gin_file = '%s.gin' % gin_file
            gin_files = [os.path.join(config_dir, gin_file)]
            model_id = os.path.split(gin_file)[1][:-4]
            gin.bind_parameter('default_model_dir.model_id', model_id)
        gin_param = FLAGS.gin_param
        if gin_param is None:
            gin_param = []
        else:
            gin_param = list(gin_param)
        gin.parse_config_files_and_bindings(gin_files, gin_param)

    fn = getattr(runners, FLAGS.command, None)
    if fn is None or not callable(fn):
        raise IOError(
            'Invalid command %s - must be a function in `ige.runners`')
    fn()


if __name__ == '__main__':
    app.run(main)
 