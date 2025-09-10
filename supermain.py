import ibmdata as encode
import worker as w
import hashing as h
import hdrfE as hd

minfile = 6
maxfile = 6
nbb = [2,4,8,16,32,64,128]

def main():
    hashtab = [[[0.0,0.0] for i in range(len(nbb))] for i in range(minfile,maxfile+2)]
    hdrfetab = [[[0.0,0.0] for i in range(len(nbb))] for i in range(minfile,maxfile+2)]
    for i in range(minfile,maxfile+1):
        for j in range(len(nbb)):

            if i <10:
                s = "0" +str(i)
            else:
                s = str(i)

            fil = "ibm"+s+".data"
            hashtab[i][j] = w.main(fil,h.chooseEdge,nbb[j])
            hdrfetab[i][j] = w.main(fil,hd.chooseEdge,nbb[j])
    with open("buff.out",'a') as f:
        for i in range(minfile,maxfile+1):
            for j in range(len(nbb)):
                f.write("file IBM" + str(i)+" nbb:"+str(j) + "\n")
                f.write("   Hash: " + str(hashtab[i])+"\n")
                f.write("   HDRF: " + str(hdrfetab[i])+"\n")

if __name__ == '__main__':
    main()
