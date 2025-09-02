#!/usr/bin/env python3
import os
import sys

def find_min_files(root_dir, start=1, end=148092):
    min_count = None
    min_dir = None

    for i in range(start, end + 1):
        if i % 1000 == 0:
            print(f"Checking directory {i}...")
        subdir = os.path.join(root_dir, str(i))
        if os.path.isdir(subdir):
            # csak közvetlen fájlok megszámolása
            count = sum(1 for f in os.listdir(subdir)
                        if os.path.isfile(os.path.join(subdir, f)))
            if min_count is None or count < min_count:
                min_count = count
                min_dir = subdir

    return min_dir, min_count

if __name__ == "__main__":
    root = "/Users/levente/Documents/SZTAKI/jester_dataset/jester"

    if not os.path.isdir(root):
        print(f"Hiba: {root} nem létező könyvtár.")
        sys.exit(1)

    min_dir, min_count = find_min_files(root)

    if min_dir is None:
        print("Nem találtunk megfelelő számozott alkönyvtárakat.")
    else:
        print(f"A legkevesebb fájl a(z): {min_dir} mappában található.")
        print(f"Fájlszám: {min_count}")
