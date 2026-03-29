#!/usr/bin/env bash

# Copyright 2024, Technical University of Munich
# SPDX-License-Identifier: BSL-1.0
# Author: Mateo de Mayo <mateo.demayo@tum.de>

git diff main --name-only | xargs clangd-tidy -p build -j $(nproc) --github --color always
