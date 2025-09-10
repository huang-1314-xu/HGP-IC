import os
import sys
import requests
import zipfile

THEME = 7
LOCATION = 9
PERSON = 11
ORGANISATION = 13

FILELOC = "masterfilelist-translation.txt"

def adding(buff,l):
    if l != []:
        buff += l

def writing(g,line):
    if line != []:
        g.write(str(line)+"\n")

def link(i):
    with open(FILELOC,'r') as f:
        l = f.readlines()
        return l[i*3+2].split()[2]

def download(link,name):
    print("Working on "+link)
    r = requests.get(link)
    with open("temp.zip",'wb') as f:
        f.write(r.content)
    zip = zipfile.ZipFile(r'temp.zip')
    zip.extractall("raw data/" + name)
    return name + "/" + os.listdir("raw data/" + name)[0]

def correct():
    #remove []
    with open("data/gdelt.data",'r') as f:
        with open("data/gdelt.data2",'w') as g:
            for line in f:
                print(line)
                if len(line) > 3:
                    g.write(line)

def getnumberline(path):
    with open("raw data/" + path,'r') as f:
        for i,line in enumerate(f):
            ()
    return(i)



def main(file,rule = 'a'):
    jtemp = 0
    with open("raw data/" + file,'r') as f:
        with open("data/gdelt.data",rule) as g:
            for i,line in enumerate(f):
                buff = []
                try:
                    adding(buff,line.split("\t")[THEME].split(";")[:-1])
                except:
                    print("ERROR on line: " + str(i) + " of file: " + file)
                try:
                    adding(buff,line.split("\t")[LOCATION].split(";")[:-1])
                except:
                    print("ERROR on line: " + str(i) + " of file: " + file)
                try:
                    adding(buff,line.split("\t")[PERSON].split(";")[:-1])
                except:
                    print("ERROR on line: " + str(i) + " of file: " + file)
                try:
                    adding(buff,line.split("\t")[ORGANISATION].split(";")[:-1])
                except:
                    print("ERROR on line: " + str(i) + " of file: " + file)
                writing(g,buff)
                jtemp += 1

    return jtemp




if __name__ == "__main__":
    i = 0
    j = 0
    while(j < 1000000):
        namei = "gdelt" + str(i)
        if not os.path.isdir("raw data/" + namei):
            linki = link(i)
            ni = download(linki,namei)
            j += main(ni,'a')
            print("work done for " + str(i))
        else:
            ni = namei + "/" + os.listdir("raw data/" + namei)[0]
            print("work skip for " + str(i))
            j += getnumberline(ni)
        print("Actually there is " + str(j) + " edges")
        i += 1
