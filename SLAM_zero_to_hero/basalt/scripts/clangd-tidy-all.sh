#!/usr/bin/env bash

# Copyright 2024, Technical University of Munich
# SPDX-License-Identifier: BSL-1.0
# Author: Mateo de Mayo <mateo.demayo@tum.de>

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
FOLDER="${1:-$SCRIPT_DIR/../include $SCRIPT_DIR/../src $SCRIPT_DIR/../test/src}"
find $FOLDER -iname "*.?pp" -or -iname "*.h" | xargs clangd-tidy -p build -j $(nproc) --github --color always
