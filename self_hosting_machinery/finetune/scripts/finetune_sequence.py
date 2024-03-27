import os
import signal
import subprocess
import sys
import psutil


def catch_sigusr1(signum, frame):
    print("catched SIGUSR1")
    current_process = psutil.Process()
    for child in current_process.children(recursive=False):
        os.kill(child.pid, signal.SIGUSR1)


def main(finetune_train_script: str):
    signal.signal(signal.SIGUSR1, catch_sigusr1)
    try:
        cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "")
        num_gpus = len(cuda_visible_devices.split(","))
        cmd = [sys.executable]
        if num_gpus > 1:
            # this tries to run the same as:
            # CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 finetune_train.py --pname project ...
            port = 20000 + hash(cuda_visible_devices) % 1000
            cmd = ["torchrun", "--master-port", str(port), f"--nproc_per_node={num_gpus}"]
        subprocess.check_call([*cmd, finetune_train_script, *sys.argv[1:]])
    except subprocess.CalledProcessError as e:
        print("finetune_sequence: %s" % e)
        sys.exit(1)


if __name__ == '__main__':
    main(os.path.join(os.path.dirname(os.path.realpath(__file__)), "finetune_train.py"))
