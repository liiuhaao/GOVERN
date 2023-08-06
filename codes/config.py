import argparse
import os
import time


def create_if_not_exit(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, 0o777)


parser = argparse.ArgumentParser(description="")
parser.add_argument("--dataset", default="all", type=str)
parser.add_argument('--seeds', default=[42, 43, 44, 45, 46], type=int, nargs='+')
parser.add_argument(
    "--name", default=time.strftime("%y%m%d%H%M%S", time.localtime()), type=str
)
args = parser.parse_args()

output_path = os.path.join("..", "outputs")
create_if_not_exit(output_path)

data_path = os.path.join("..", "datasets")
datasets_dict = {data: f"{data}/" for data in os.listdir(data_path)}
