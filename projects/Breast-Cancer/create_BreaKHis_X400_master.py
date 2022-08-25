# Copyright 2022 (C) antillia.com. All Rights Reserved.
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

# 2022/08/26 Copyright (C) antillia.com

# create_BreaKHis_X400_master.py

import os
import sys
import glob
import shutil
import traceback


def create_BreaKHis_X400_master(source_dir, dest_dir):
  if os.path.exists(dest_dir):
    shuti.rmtree(dest_dir)
  if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
  print("--- create_BreaKHis_X400_master {} {}".format(source_dir, dest_dir))
  pattern = source_dir + "/SOB/*/*/400X/*.png"
  files = glob.glob(pattern)
  n = 1
  for file in files: 
    shutil.copy2(file, dest_dir)
    print("---{}  copied {} to {}".format(n, file, dest_dir))
    n += 1
  
if __name__ == "__main__":
  sub_dirs       = ["benign", "malignant"]
  source_top_dir = "D:/BreaKHis_v1/histology_slides/breast/"
  dest_top_dir   = "c:/work/BreaKHvis_v1_400X/master"
  if len(sys.argv) == 3:
    source_top_dir = sys.argv[1]
    dest_top_dir   = sys.argv[2]
  
  if not os.path.exists(source_top_dir):
    raise Exception("Not found source_top_dir " + source_top_dir)

  if not os.path.exists(dest_top_dir):
    os.makedirs(dest_top_dir)
  print("source_top_dir  {}".format(source_top_dir))
  print("dest_top_dir    {}".format(dest_top_dir))
  input("Hit any key to start")

  try:
    for sub_dir in sub_dirs:
      source_dir = os.path.join(source_top_dir, sub_dir)
      if not os.path.exists(source_dir):
        raise Exception("Invalid source_dir " + source_dir)

      dest_dir   = os.path.join(dest_top_dir, sub_dir)
      create_BreaKHis_X400_master(source_dir, dest_dir)

  except:
    traceback.print_exc()

    