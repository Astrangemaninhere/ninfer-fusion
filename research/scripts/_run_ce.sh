#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
bash -n "$J/_chunk_equivalence.sh" && echo SYNTAX_OK || exit 1
bash "$J/_chunk_equivalence.sh" 2>&1 | tail -18 | cut -c1-200
