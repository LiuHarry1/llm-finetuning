import sys

for line in sys.stdin:
    #a = line.split()

    #print(int(a[0]) + int(a[1]))

    c_count =0
    ch_count =0
    chn_count = 0
    s= line
    for ch in s:
        if ch =='C':
            c_count +=1
        elif ch=='H':
            ch_count +=c_count
        elif ch =='N':
            chn_count +=ch_count

    print(chn_count)


