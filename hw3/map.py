import sys

d = {}

with open(sys.argv[1], encoding='big5hkscs') as fd:
    for line in fd:
        line = line.split(' ')
        big5 = line[0]
        zhuyin = line[1].strip()
        zhuyin = zhuyin.split('/')
        for c in zhuyin:
            c = c[0]
            if c not in d:
                d[c] = set()
            d[c].add(big5)
        d[big5] = [big5]

with open(sys.argv[2], 'w+', encoding='big5hkscs') as fd:
    for k, v in d.items():
        print('{}\t{}'.format(k, ' '.join(list(v))), file=fd)
