# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import pathlib
import subprocess
import sys


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime_only",
        help=(
            "Only check the initialization of the submodules for the"
            "runtime-dependent submodules. Default: False"
        ),
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    # No-op if we're not in a git repository.
    try:
        subprocess.check_call(
            ["git", "rev-parse", "--is-inside-work-tree"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except:
        return

    repo_root = pathlib.Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    )
    modules_file = repo_root / ".gitmodules"
    if not modules_file.exists():
        return
    paths = subprocess.check_output(
        ["git", "config", "--file", str(modules_file), "--get-regexp", "path"],
        text=True,
    ).splitlines()
    submodules = [line.split(maxsplit=1)[1] for line in paths]

    runtime_submodules = (
        pathlib.Path(__file__)
        .with_name("runtime_submodules.txt")
        .read_text()
        .split("\n")
    )

    for name in submodules:
        if (not args.runtime_only or name in runtime_submodules) and not (
            repo_root / name / ".git"
        ).exists():
            print(
                "The git submodule '%s' is not initialized. Please run `git submodule update --init`"
                % (name)
            )
            sys.exit(1)


if __name__ == "__main__":
    run()
