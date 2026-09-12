#!/bin/bash
A=/home/user/ninfer-fusion
echo "=== newest 60 source files by mtime in tree:"
find "$A/src" "$A/include" "$A/apps" -type f \( -name '*.h' -o -name '*.cuh' -o -name '*.cu' -o -name '*.cpp' \) -printf '%TY-%Tm-%Td %TH:%TM  %p\n' 2>/dev/null | sort -r | head -60
