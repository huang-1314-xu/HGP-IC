import matplotlib.pyplot as plt
import sys

def stats(path):
    t = []
    lentab = []
    with open("data/"+path,'r') as f:
        for i,line in enumerate(f):
            l = eval(line)
            if len(l) >= 2:
                t.append(l)
                lentab.append(len(l))
    v = dict()
    with open("data/"+path,'r') as f:
        for line in f:
            vertices = eval(line)
            for vertice in vertices:
                if v.has_key(vertice):
                    v[vertice] += 1
                else:
                    v[vertice] = 1
    t = v.values()
    t.sort()
    lentab.sort()
    moy = sum(lentab)/float(i)
    med = lentab[len(lentab)/2]
    moyt = sum(t)/float(i)
    medt = lentab[len(t)/2]
    print("Number of line of file: "+ path + " :" + str(i))
    print("Average edge length:" + str(moy))
    print("Mediane edge length:" + str(med))
    print("Max edge length:" + str(lentab[-1]))
    print("Average vertices appearance:" + str(moyt))
    print("Mediane vertices appearance:" + str(medt))
    print("Max vertices length:" + str(t[-1]))


def lenofedges(path):
    t = []
    with open("data/"+path,'r') as f:
        for i,line in enumerate(f):
            l = eval(line)
            if len(l) >= 2:
                t.append(len(l))
        print("Number of line of file: "+ path + " :" + str(i))
    s = [0 for i in range(max(t)+1)]
    for i in t:
        s[i] += 1
    return(s)

def lenofvertices(path):
    v = dict()
    with open("data/"+path,'r') as f:
        for line in f:
            vertices = eval(line)
            for vertice in vertices:
                if v.has_key(vertice):
                    v[vertice] += 1
                else:
                    v[vertice] = 1
    t = v.values()
    s = [0 for i in range(max(t)+1)]
    for i in t:
        s[i] += 1
    print(s)
    return(s)

def lenofsameedge(path):
    v = dict()
    with open("data/"+path,'r') as f:
        for line in f:
            if v.has_key(line):
                v[line] += 1
            else:
                v[line] = 1
    t = v.values()
    s = [0 for i in range(max(t)+1)]
    for i in t:
        s[i] += 1
    print(s)
    return(s)

def extensivetest(path):
    with open("data/"+path,'r') as f:
        for i,line in enumerate(f):
            if line == "['TAX_POLITICAL_PARTY', 'TAX_POLITICAL_PARTY_CONSERVATIVE_PARTY', 'TAX_FNCACT', 'TAX_FNCACT_WORKERS', 'TAX_POLITICAL_PARTY_WORKERS_PARTY', 'DEMOCRACY', 'peoples party', 'national rights', 'party in furstentum liechtenstein']\n":
                print(i)

if __name__ == '__main__':
    fig, ax = plt.subplots()
    s0 = lenofsameedge("gdelt.data")
    s1 = lenofsameedge("gdeltnoD.data")
    s2 = lenofsameedge("ibm18.data")
    plt.loglog(s0)
    plt.loglog(s1)
    plt.loglog(s2)
    plt.legend(["GDELT","GDELT no duplicate","IBM18"])
    plt.ylabel("Number of unique edge")
    plt.xlabel("Number of copies of an unique edge")
    plt.title('Same edge distribution')
    plt.show()
    s0 = lenofedges("gdelt.data")
    s1 = lenofedges("gdeltnoD.data")
    s2 = lenofedges("ibm18.data")
    plt.semilogy(s0)
    plt.semilogy(s1)
    plt.semilogy(s2)
    plt.legend(["GDELT","GDELT no duplicate","IBM18"])
    plt.xlabel("Edge degree")
    plt.ylabel("Number of edge")
    plt.title("Edge degree distribution")
    plt.show()
    e0 = lenofvertices("gdelt.data")
    e1 = lenofvertices("gdeltnoD.data")
    e2 = lenofvertices("ibm18.data")
    plt.figure(1)
    plt.loglog(e0)
    plt.loglog(e1)
    plt.loglog(e2)
    plt.legend(["GDELT","GDELT no duplicate","IBM18"])
    plt.xlabel("Vertex degree")
    plt.ylabel("Number of vertex")
    plt.title('Vertex degree distribution')
    plt.show()

# if __name__ == '__main__':
#     stats("ibm18.data")
#     stats("gdeltnoD.data")
