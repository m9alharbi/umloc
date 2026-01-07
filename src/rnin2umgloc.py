#!/usr/bin/env python
# rename_global_write_lists.py
#
# root/
# ├─ data_train/
# ├─ data_val/
# └─ data_test/
#
# After running:
#   every seq dir is named 0,1,2… with no duplicates,
#   and each split has a *.txt list of its own IDs.

import os, argparse

def rename_and_list(root, split_order=("data_train", "data_val", "data_test")):
    global_id   = 0
    list_files  = {"data_train": "train.txt",
                   "data_val"  : "val.txt",
                   "data_test" : "test.txt"}

    for split in split_order:
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            print(f"[SKIP] {split_dir} (missing)")
            continue

        # original folders
        orig_dirs = sorted([d for d in os.listdir(split_dir)
                            if os.path.isdir(os.path.join(split_dir, d))])
        if not orig_dirs:
            print(f"[WARN] {split_dir} empty")
            continue

        # pass 1: rename to temporary names to avoid collisions
        for i, d in enumerate(orig_dirs):
            os.rename(os.path.join(split_dir, d),
                      os.path.join(split_dir, f"__tmp_{i}"))

        # pass 2: rename to global incremental IDs & collect names
        final_names = []
        tmp_dirs = sorted([d for d in os.listdir(split_dir) if d.startswith("__tmp_")])
        for tmp in tmp_dirs:
            src = os.path.join(split_dir, tmp)
            dst = os.path.join(split_dir, str(global_id))
            os.rename(src, dst)
            final_names.append(str(global_id))
            global_id += 1

        # write list file
        list_path = os.path.join(split_dir, list_files[split])
        with open(list_path, "w") as f:
            f.write("\n".join(final_names))
        print(f"[OK] {split_dir}: renamed {len(final_names)} seqs, list saved to {list_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root",
        help="dataset root containing data_train/ data_val/ data_test/")
    args = parser.parse_args()
    rename_and_list(args.root)
