import pickle

def run(chooser,nbrOfPart,stream,vPP,graph):
    comp = []
    #run a partitions streaming chooser on file of hyperEdge
    partitions = [[]for i in range(nbrOfPart)]
    l = len(stream)
    for i,hedge in enumerate(stream):
        pos = chooser(hedge,partitions,vPP,graph)
        partitions[pos].append(hedge)
        for vertex in hedge:
            vPP[pos].add(vertex)
        if i%1000 == 0:
            print("Percent done: " + str(float(i)/l*100))
    with open("/23085411005/hdrf_hy_pycharm/vertex_partitions/hdrf/cora/2", 'wb') as f:
        pickle.dump((partitions), f)
    return(partitions)
