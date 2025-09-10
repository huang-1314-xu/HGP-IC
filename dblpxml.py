from lxml import etree

FILENAME = "dblp.xml"

def elem2dict(node):
    """
    Convert an lxml.etree node tree into a dict.
    """
    d = {}
    for e in node.iterchildren():
        key = e.tag.split('}')[1] if '}' in e.tag else e.tag
        value = e.text if e.text else elem2dict(e)
        d[key] = value
    return d

def main():
    with open("raw data/" + FILENAME) as f:
        s = f.read()
    node = etree.fromstring(s)
    d = elem2dict(node)
    print(d)

if __name__ == '__main__':
    main()
