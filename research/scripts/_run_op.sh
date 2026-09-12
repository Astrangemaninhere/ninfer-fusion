#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
bash -n "$J/_offset_probe.sh" && echo SYNTAX_OK || exit 1
bash "$J/_offset_probe.sh" 2>&1 | tail -20 | cut -c1-200
