import argparse
import os

import gsd
import gsd.hoomd

# with gsd.hoomd.open("../scripts/slj.hairy.gsd", "rb") as traj, gsd.hoomd.open("../scripts/slj.stride.gsd", "wb") as new_traj:
#     print(len(traj))
#     for frame in range(0, len(traj), 50):
#         s = gsd.hoomd.Snapshot()
#         s = traj[frame]
#         new_traj.append(s)

def main(input_file, output_file):
    with gsd.hoomd.open(input_file, "rb") as in_traj, \
         gsd.hoomd.open(output_file, "wb") as out_traj:
        # s = gsd.hoomd.Snapshot()
        s = in_traj[-1]
        out_traj.append(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="grab the last frame of a gsd traj and save")
    parser.add_argument("input", type=str)
    parser.add_argument("--output", "-o", type=str)
    args = parser.parse_args()

    # error checking
    if not os.path.exists(args.input):
        raise RuntimeError("Spec'd input file does not exist!")
    input_file = args.input
    print(os.path.splitext(input_file))

    if args.output is None:
        fname = os.path.splitext(input_file)[0]
        ext = os.path.splitext(input_file)[-1]
        args.output = f"{fname}.out{ext}"
    output_file = args.output
    if os.path.exists(output_file):
        raise RuntimeError("Aborting: would overwrite existing file!")
    main(input_file, output_file)
