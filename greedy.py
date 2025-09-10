eps = 1
rho = 1.1
maxsize = 0
minsize = 0
lamb = 1.01
partdeg = {}
s = 0

def CgreedyBal(part):
    n = maxsize - len(part)
    d = eps + maxsize - minsize
    return (n/d)

def CHDRFeREP(hedge,vpp,i):
    s = 0.0
    for vertex in hedge:
        if vertex in vpp[i]:
            s = 1
    return s

def CHDRFe(hedge,part,vpp,i):
    part1 = CHDRFeREP(hedge,vpp,i)
    part2 = CgreedyBal(part)
    return part1 + lamb*part2


def init(vertexs,numberOfPartitions,stream):
    global maxsize
    global minsize
    card = len(stream)
    s = float(card)/numberOfPartitions
    maxsize = int(rho*s)
    minsize = int((1/rho)*s)

def chooseEdge(hedge,partitions,vpp,_eps=1.1,_rho=1.1):
    #return a the number of the partition where edge should go
    global eps
    global rho
    eps = _eps
    rho = _rho
    buff = -1
    scoreBuff = -1
    for i,part in enumerate(partitions):
        scoreloc = CHDRFe(hedge,part,vpp,i)
        if scoreloc > scoreBuff:
            buff = i
            scoreBuff = scoreloc
    return buff
