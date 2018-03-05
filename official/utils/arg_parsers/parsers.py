# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

import argparse


class LocationParser(argparse.ArgumentParser):
  def __init__(self, add_help=False):
    super(LocationParser, self).__init__(add_help=add_help)
    self.add_argument(
        "--data_dir", "-dd", default="/tmp",
        help="[default: %(default)s] The location of the input data.",
        metavar="<DD>",
    )

    self.add_argument(
        "--model_dir", "-md", default="/tmp",
        help="[default: %(default)s] The location of the model files.",
        metavar="<MD>",
    )


class SupervisedParser(argparse.ArgumentParser):
  def __init__(self, add_help=False):
    super(SupervisedParser, self).__init__(add_help=add_help)
    self.add_argument(
        "--train_epochs", "-te", type=int, default=1,
        help="[default: %(default)s] The number of epochs to use for training.",
        metavar="<TE>"
    )

    self.add_argument(
        "--epochs_per_eval", "-epe", type=int, default=1,
        help="[default: %(default)s] The number of training epochs to run "
             "between evaluations.",
        metavar="<EPE>"
    )

    self.add_argument(
        "--batch_size", "-bs", type=int, default=32,
        help="[default: %(default)s] Batch size for training and evaluation.",
        metavar="<BS>"
    )


class PerformanceParser(argparse.ArgumentParser):
  def __init__(self, add_help=False):
    super(PerformanceParser, self).__init__(add_help=add_help)
    self.add_argument(
        "--num_parallel_calls", "-npc",
        type=int, default=5,
        help="[default: %(default)s] The number of records that are processed "
             "in parallel  during input processing. This can be optimized per "
             "data set but for generally homogeneous data sets, should be "
             "approximately the number of available CPU cores.",
        metavar="<NPC>"
    )

    self.add_argument(
        "--inter_op_parallelism_threads", "-inrt",
        type=int, default=0,
        help="[default: %(default)s Number of inter_op_parallelism_threads to "
             "use for CPU. See TensorFlow config.proto for details.",
        metavar="<INRT>"
    )

    self.add_argument(
        "--intra_op_parallelism_threads", "-inat",
        type=int, default=0,
        help="[default: %(default)s Number of intra_op_parallelism_threads to "
             "use for CPU. See TensorFlow config.proto for details.",
        metavar="<INAT>"

    )


class DummyParser(argparse.ArgumentParser):
  def __init__(self, add_help=False):
    super(DummyParser, self).__init__(add_help=add_help)
    self.add_argument(
        "--use_synthetic_data", "-usd",
        action="store_true",
        help="If set, use fake data (zeroes) instead of a real dataset. "
             "This mode is useful for performance debugging, as it removes "
             "input processing steps, but will not learn anything."
    )


class ImageModelParser(argparse.ArgumentParser):
  def __init__(self, add_help=False):
    super(ImageModelParser, self).__init__(add_help=add_help)
    self.add_argument(
        "--data_format", "-df",
        help="A flag to override the data format used in the model. "
             "channels_first provides a performance boost on GPU but is not "
             "always compatible with CPU. If left unspecified, the data format "
             "will be chosen automatically based on whether TensorFlow was "
             "built for CPU or GPU.",
        metavar="<CF>",
    )


class ProtoResNetParser(argparse.ArgumentParser):
  """
  Just to make sure inheritance is working correctly. Will be removed.
  """
  def __init__(self):
    super(ProtoResNetParser, self).__init__(parents=[
      LocationParser(),
      SupervisedParser(),
      ImageModelParser(),
      PerformanceParser(),
      DummyParser(),
    ])

parser = ProtoResNetParser()
args = parser.parse_args(["-h"])


