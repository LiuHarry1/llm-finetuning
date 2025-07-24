import sys
import re

"""
5
A123
1ABC
A1B2C3
123ABC
A!23
"""

def package_cat(pkg_id):
    pkg_id = pkg_id.replace("\n", "")
    print(pkg_id)
    if re.fullmatch(r'[A-Za-z]+[0-9]+', pkg_id):
        return "standard"
    elif re.fullmatch(r'[0-9]+[A-Za-z]+', pkg_id):
        return "speical"
    elif re.fullmatch(r'[A-Za-z]+[A-Za-z0-9]+', pkg_id):
        return "mix"
    else:
        return "invalid"

s = sys.stdin.readline()
n = int(s)

for i in  range(n):
    package_id = sys.stdin.readline()
    print(package_cat(package_id))






