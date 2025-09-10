# from sets import Set

def uniqueVertex(path):
    with open("data/" + path,'r') as f:
        s = {}
        for line in f:
            hedge = eval(line)
            for vertex in hedge:
                if vertex in s:
                    s[vertex] += 1
                else:
                    s[vertex] = 1
        print(len(s.values()))

if __name__ == '__main__':
    uniqueVertex("ibm18.data")
