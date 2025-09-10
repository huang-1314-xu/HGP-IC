import math

def replicationFactor(partitions,listofVertex,vpp):
    s = 0.0
    numberofEdge = 0.0
    for part in partitions:
        numberofEdge += len(part)
    l = len(listofVertex)
    for i,vertex in enumerate(listofVertex):
        for j in range(len(partitions)):
            if vertex in vpp[j]:
                s += 1
        if i % 100 == 0:
            print("Score is at " + str(float(i)/l*100) + " percent")
    s = s / len(listofVertex)
    return s

def RF_RSD(partitions,listofVertex,vpp):
    s = 0.0
    numberofEdge = 0.0
    for part in partitions:
        numberofEdge += len(part)
    l = len(listofVertex)
    for i,vertex in enumerate(listofVertex):
        for j in range(len(partitions)):
            if vertex in vpp[j]:
                s += 1
    s = s / len(listofVertex)
    S = 0.0
    xbarre = numberofEdge/len(partitions)
    for part in partitions:
        x = len(part)
        S += (x - xbarre) * (x - xbarre)
    S = S / len(partitions)
    S = math.sqrt(S) * 100 / xbarre
    S2 = 0
    for part in partitions:
        S2 = max(S2,len(part))
    S2 = (S2 - int(xbarre))
    return (s,S,S2)
