import sys

def writing(g,line):
    g.write(str(line)+"\n")

def main(file):
    with open("raw data/" + file,'r') as f:
        with open("data/"+file[:-8]+".data",'w') as g:
            buff = []
            for i,line in enumerate(f):
                if i >= 5:
                    l = line.split(" ")
                    if l[1] == "s" and buff != []:
                        writing(g,buff)
                        buff = [l[0]]
                    else:
                        buff.append(l[0])


if __name__ == "__main__":
    main(sys.argv[1])
