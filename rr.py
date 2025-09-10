def chooseEdge(edge,partitions,vpp):
    #return a the number of the partition where edge should go
    index = -1
    m = 9999999
    for i,part in enumerate(partitions):
        if len(part) < m:
            index = i
            m = len(part)
    return index
