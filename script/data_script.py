import xml.etree.ElementTree as ET
import pickle, re, sys, os

def read_file(xml_path, text_path, event_type):
    apf_tree = ET.parse(xml_path)
    root = apf_tree.getroot()
    
    event_start = {}
    event_end = {}

    event_ident = {}
    event_map = {}
    event = dict()

    for events in root.iter("event"):
        ev_type = events.attrib["TYPE"] + "_" + events.attrib["SUBTYPE"]
        if ev_type not in event_type:
            event_type.append(ev_type)
        for mention in events.iter("event_mention"):
            ev_id = mention.attrib["ID"]
            anchor = mention.find("anchor")
            for charseq in anchor:
                start = int(charseq.attrib["START"])
                end = int(charseq.attrib["END"]) + 1
                text = re.sub(r"\n", r"", charseq.text)
                event_tupple = (ev_type, start, end, text)
                if event_tupple in event_ident:
                    sys.stderr.write("dulicapte event {}\n".format(ev_id))
                    event_map[ev_id] = event_ident[event_tupple]
                    continue
                event_ident[event_tupple] = ev_id
                event[ev_id] = [ev_id, ev_type, start, end, text]
                event_start[start] = ev_id
                event_end[end] = ev_id

    doc = open(text_path).read()
    doc = re.sub(r"<[^>]+>", r"", doc) 
    doc = re.sub(r"(\S+)\n(\S[^:])", r"\1 \2", doc)
    print(text_path)
    offset = 0
    size = len(doc)
    try:
        header, _, finish = doc.split(r"\n\n\n\n")
        current = len(header) + 4
        end = len(doc) - len(finish)
    except:
        end = len(doc)
        current = 0
    regions = []
    tokens = []
    anchors = []
    check= 0
    for i in range(size):
        if i in event_start:
            inc = 0
            new = clean_str(doc[current:i])
            regions.append(new)
            tokens += new.split()
            check = 1
            anchors += [0 for _ in range(len(new.split()))]
            inc = 0
            current = i
            ent = event_start[i]
            event[ent][2] += offset + inc
        if i in event_end:
            ent = event_end[i]
            event[ent][3] += offset
            new = clean_str(doc[event[ent][2] : event[ent][3]])
            regions.append(new)
            tokens += [new]
            anchors += [event_type.index(event[ent][1])]
            offset += inc
            current = event[ent][3] 
    new = clean_str(doc[current : end])
    regions.append(new)
    tokens += new.split()
    anchors += [0 for _ in range(len(new.split()))]
    doc = "".join(regions)
    if len(tokens) == 0:
        print(doc)
        print(text_path)
    for e in  event.values():
        if "\n" in doc[int(e[2]) : int(e[3])]:
            l = []
            l.append(doc[0 : int(e[2])])
            l.append(doc[int(e[2]) : int(e[3])].replace("\n", " "))
            l.append(doc[int(e[3]) :])
            doc = "".join(l)

    #for e in event.values():
    #    assert doc[int(e[2]):int(e[3])].replace("&AMP;","&").replace("&amp;", "&").replace(" ","") \
    #        == e[4].replace(" ",""), "%s <=> %s" % (doc[int(e[2]):int(e[3])], e[4])
        

    #sys.stdout.write(doc)
    #for i in range(len(tokens)):
    #    print(anchors[i], tokens[i])
    #for e in event.values():
    #    sys.stdout.write(str(tuple(e)) + "\n")
    return tokens, anchors

def encode_corpus(folder_path):
    file_list = os.path.join(folder_path, "FileList")
    files = []
    with open(file_list) as f:
        for line in f:
            map = line.strip().split()
            if len(map) != 3: continue
            files.append((map[0], map[1].split(",")[-1]))
    return files

def read_corpus(folder_path, event_type):
    count = 0
    file_list = encode_corpus(folder_path)
    tokens, anchors = [], []
    for (file, path) in file_list:
        file_path = os.path.join(folder_path, path, file)
        tok, anc = read_file(file_path + ".apf.xml", file_path + ".sgm", event_type)
        count += 1
        tokens.append(tok)
        anchors.append(anc)
    #print(count, len(event_type))
    return tokens, anchors

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)  
    string = re.sub(r"\'m", r" 'm", string)
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r"\.", " <dot>", string)
    string = re.sub(r"\,", r" <dot> ", string) 
    string = re.sub(r"!", " <dot> ", string) 
    string = re.sub(r"\(", " <dot> ", string) 
    string = re.sub(r"\)", " <dot> ", string) 
    string = re.sub(r"\?", " <dot> ", string) 
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

if __name__ == "__main__":
    test = r"/home/jeovach/PycharmProjects/ed_ace/ace_2005_td_v7/data/English/nw/fp2/AFP_ENG_20030630.0741"
    event_type = [None]
    #read_file(test+".apf.xml", test+".sgm", event_type = [])
    tokens, anchors  = read_corpus(
        "/home/jeovach/PycharmProjects/ed_ace/ace_2005_td_v7/data/English/bn",
        event_type)
    t, a = read_corpus(
        "/home/jeovach/PycharmProjects/ed_ace/ace_2005_td_v7/data/English/nw",
        event_type)
    tokens += t
    anchors += a
    pickle.dump(tokens, open("tokens1.bin","wb"))
    pickle.dump(anchors, open("anchors1.bin", "wb"))

    t, a = read_corpus(
        "/home/jeovach/PycharmProjects/ed_ace/ace_2005_td_v7/data/English/bc",
        event_type)
    tokens = t
    anchors = a
    pickle.dump(tokens, open("tokens2.bin","wb"))
    pickle.dump(anchors, open("anchors2.bin", "wb"))

    t, a = read_corpus(
        "/home/jeovach/PycharmProjects/ed_ace/ace_2005_td_v7/data/English/cts",
        event_type)
    tokens = t
    anchors = a
    pickle.dump(tokens, open("tokens3.bin","wb"))
    pickle.dump(anchors, open("anchors3.bin", "wb"))
    t, a = read_corpus(
        "/home/jeovach/PycharmProjects/ed_ace/ace_2005_td_v7/data/English/wl",
        event_type)
    tokens = t
    anchors = a
    print(len(event_type))
    print(event_type)
    pickle.dump(tokens, open("tokens4.bin","wb"))
    pickle.dump(anchors, open("anchors4.bin", "wb"))
