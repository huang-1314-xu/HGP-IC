import simu
import score
import hashing,hdrfE,rr,rand,greedy
import random
import sys
from sets import Set
import hdrfEplus

def openData(path):
    s = []
    with open("data/"+path,'r') as f:
        for line in f:
            hedge = eval(line)
            s.append(hedge)
    return(s)


def vertexinset(file):
    v = Set()
    with open("data/" + file,'r') as f:
        for line in f:
            hedge = eval(line)
            for vertex in hedge:
                if vertex not in v:
                    v.add(vertex)
    return v

def output(s):
    with open("data.out",'a') as f:
        for i in s:
            f.write(str(i) + " ")
        f.write("\n")

def main(testFile,algo = hashing.chooseEdge,nbb = 133):
    #Set up
    random.seed(2)
    init = hdrfE.init
    scoring = score.replicationFactor
    numberOfPartitions = nbb
    #End Set up
    vpp = [Set() for i in range(numberOfPartitions)] #Vertex per partitions
    stream = openData(testFile)
    random.shuffle(stream)
    vertexs = vertexinset(testFile)
    print("Vertex got. Total of: " + str(len(vertexs)) + "vertexs")
    init(vertexs,numberOfPartitions,stream)
    res = simu.run(algo,numberOfPartitions,stream,vpp)
    print("Simulation done, computing score:")
    rf = score.replicationFactor(res,vertexs,vpp)
    print("Replication factor is: " + str(rf))
    return rf

def testNumPart(testFile,algo,nbb):
    #Set up
    random.seed(2)
    init = hdrfE.init
    scoring = score.replicationFactor
    #End Set up
    sc = []
    for numberOfPartitions in nbb:
        print(numberOfPartitions)
        vpp = [Set() for i in range(numberOfPartitions)] #Vertex per partitions
        stream = openData(testFile)
        random.shuffle(stream)
        vertexs = vertexinset(testFile)
        print("Vertex got. Total of: " + str(len(vertexs)) + "vertexs")
        init(vertexs,numberOfPartitions,stream)
        res = simu.run(algo,numberOfPartitions,stream,vpp)
        print("Simulation done, computing score:")
        sc.append(scoring(res,vertexs,vpp))
        print("Replication factor is: " + str(sc))
    return sc

def mainEvery(testFile,algo = hashing.chooseEdge,nbb = 133):
    #Set up
    random.seed(2)
    init = hdrfE.init
    scoring = score.RF_RSD
    numberOfPartitions = nbb
    #End Set up
    vpp = [Set() for i in range(numberOfPartitions)] #Vertex per partitions
    stream = openData(testFile)
    random.shuffle(stream)
    vertexs = vertexinset(testFile)
    init(vertexs,numberOfPartitions,stream)
    res = simu.run(algo,numberOfPartitions,stream,vpp)
    sS = scoring(res,vertexs,vpp)
    print("State of the partitions")
    print("Edge / |Partitions|: " + str(len(stream)/nbb))
    print("Deviation of each partition")
    for part in res:
        print(len(part)-len(stream)/nbb)
    return sS


if __name__ == '__main__':
    #s = main("gdelt.data",hashing.chooseEdge)
    path = "ibm18.data"
    print("Working on: " + path)
    # print("order: random,hashing,rr,hdrfE")
    # s = mainEvery(path,rand.chooseEdge,133)
    # print(s)
    # output(s)
    # s1 = mainEvery(path,hashing.chooseEdge,133)
    # print(s1)
    # output(s1)
    # s2 = mainEvery(path,rr.chooseEdge,133)
    # print(s2)
    # output(s2)
    s3 = mainEvery(path,hdrfE.chooseEdge,133)
    print(s3)
    output(s3)
