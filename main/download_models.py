import lgp.paths
import signal
import os
import subprocess
import sys


def signal_handler(sig, frame):
    print('Stopping...')
    sys.exit(0)


if __name__ == "__main__":
    model_dir = lgp.paths.get_model_dir()
    os.makedirs(model_dir, exist_ok=True)
    signal.signal(signal.SIGINT, signal_handler)

    # Ask for verification
    print(f"Download models to: {model_dir}?")
    while True:
        x = input("y/n?>")
        if x in ["n", "no"]:
            print("Stopping. Edit init.sh and change WS_DIR or LGP_MODEL_DIR if needed.")
            sys.exit(0)
        elif x in ["y"]:
            break
        else:
            print(f"Not understood: {x}")

    print("Downloading high-level controller...")
    subprocess.call(f"wget -O '{model_dir}/alfred_hlsm_subgoal_model_e5.pytorch' 'https://github.com/hlsm-alfred/hlsm-alfred.github.io/blob/main/docs/models/alfred_hlsm_subgoal_model_e5.pytorch?raw=true'", shell=True)

    print("Downloading low-level navigation model...")
    subprocess.call(f"wget -O '{model_dir}/hlsm_gofor_navigation_model_e5.pytorch' 'https://github.com/hlsm-alfred/hlsm-alfred.github.io/blob/main/docs/models/hlsm_gofor_navigation_model_e5.pytorch?raw=true'", shell=True)

    print("Downloading depth model...")
    subprocess.call(f"wget -O '{model_dir}/hlsm_depth_model_e3.pytorch' 'https://github.com/hlsm-alfred/hlsm-alfred.github.io/blob/main/docs/models/hlsm_depth_model_e3.pytorch?raw=true'", shell=True)

    print("Downloading segmentation model...")
    subprocess.call(f"wget -O '{model_dir}/hlsm_segmentation_model_e4.pytorch' 'https://github.com/hlsm-alfred/hlsm-alfred.github.io/blob/main/docs/models/hlsm_segmentation_model_e4.pytorch?raw=true'", shell=True)

    print("Done!")
    print("You can run the evaluation scripts now.")
