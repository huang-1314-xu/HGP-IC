from sets import Set

def removeduplicate():
    with open("data/gdelt.data",'r') as f:
        with open("data/gdeltnoD.data",'w') as g:
            s = Set()
            for line in f:
                if line not in s:
                    s.add(line)
                    g.write(line)

removeduplicate()
