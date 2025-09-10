import random

def chooseEdge(edge,partitions,vpp):
    #return a the number of the partition where edge should go
    return random.randint(0,len(partitions)-1)
