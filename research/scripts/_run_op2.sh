#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
bash -n "$J/_offset_probe2.sh" && echo SYNTAX_OK || exit 1
bash "$J/_offset_probe2.sh" 2>&1 | tail -16 | cut -c1-190
