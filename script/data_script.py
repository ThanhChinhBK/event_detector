import xml.etree.ElementTree as ET
import pickle, re, sys, os
from lxml import etree
import xml_parse
from xml.dom import minidom

EVENT_MAP={'None': 0, 'Personnel.Nominate': 1, 'Contact.Phone-Write': 27, 'Business.Declare-Bankruptcy': 3,
           'Justice.Release-Parole': 4, 'Justice.Extradite': 5, 'Personnel.Start-Position': 22,
           'Justice.Fine': 7, 'Transaction.Transfer-Money': 8, 'Personnel.End-Position': 9,
           'Justice.Acquit': 10, 'Life.Injure': 11, 'Conflict.Attack': 12, 'Justice.Arrest-Jail': 13,
           'Justice.Pardon': 14, 'Justice.Charge-Indict': 15, 'Conflict.Demonstrate': 16,
           'Contact.Meet': 17, 'Business.End-Org': 18, 'Life.Be-Born': 19, 'Personnel.Elect': 20, 
           'Justice.Trial-Hearing': 21, 'Life.Divorce': 6, 'Justice.Sue': 23, 'Justice.Appeal': 24,
           'Business.Merge-Org': 32, 'Life.Die': 26, 'Business.Start-Org': 2, 'Justice.Convict': 28,
           'Movement.Transport': 29, 'Life.Marry': 30, 'UNKOWN': 34, 'Justice.Sentence': 31,
           'Justice.Execute': 25, 'Transaction.Transfer-Ownership': 33}

ace_path = "ace_2005_td_v7/data/English/"

def read_file(xml_path, text_path):
    print(text_path)
    apf_tree = ET.parse(xml_path)
    root = apf_tree.getroot()
    
    event_start = {}
    event_end = {}

    event_ident = {}
    event_map = {}
    event = dict()

    for events in root.iter("event"):
        ev_type = events.attrib["TYPE"] + "." + events.attrib["SUBTYPE"]
        for mention in events.iter("event_mention"):
            ev_id = mention.attrib["ID"]
            anchor = mention.find("anchor")
            for charseq in anchor:
                start = int(charseq.attrib["START"])
                end = int(charseq.attrib["END"]) + 2
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

    if "bn" in text_path:
        text = ""
        try:
            doc = minidom.parse(text_path)
        except:
            print("akjfjkadsbfjksdbakjfnasdjfnasdk",text_path)
        doc_root = doc.documentElement
        turn_nodes = xml_parse.get_xmlnode(doc_root, "TURN")
        for turn_node in turn_nodes:
            text += " " + xml_parse.get_nodevalue(turn_node, 0).replace("\n", " ")
        doc_id = xml_parse.get_nodevalue(
            xml_parse.get_xmlnode(doc_root,
            "DOCID")[0])
        doc_type = xml_parse.get_nodevalue(
            xml_parse.get_xmlnode(doc_root,
            "DOCTYPE")[0])
        date_time = xml_parse.get_nodevalue(
            xml_parse.get_xmlnode(doc_root,
            "DATETIME")[0])
    
        #print(text)
        sub = len(doc_id) + len(doc_type) + len(date_time) + 6
        tokens, anchors = read_document(text, sub,event_start, 
                                        event_end, event_ident, event_map, event)
 
        return [tokens], [anchors]

    if "bc" in text_path:
        text = ""
        try:
            doc = minidom.parse(text_path)
        except:
            print("akjfjkadsbfjksdbakjfnasdjfnasdk",text_path)
        doc_root = doc.documentElement
        turn_nodes = xml_parse.get_xmlnode(doc_root, "TURN")
        for turn_node in turn_nodes:
            text += " " + xml_parse.get_nodevalue(turn_node, 0).replace("\n", " ")
        doc_id = xml_parse.get_nodevalue(
            xml_parse.get_xmlnode(doc_root,
            "DOCID")[0])
        doc_type = xml_parse.get_nodevalue(
            xml_parse.get_xmlnode(doc_root,
            "DOCTYPE")[0])
        date_time = xml_parse.get_nodevalue(
            xml_parse.get_xmlnode(doc_root,
            "DATETIME")[0])
    
        #print(text)
        sub = len(doc_id) + len(doc_type) + len(date_time) + 6
        tokens, anchors = read_document(text, sub,event_start, 
                                        event_end, event_ident, event_map, event)
 
        return [tokens], [anchors]
 
    elif "nw" in text_path or "GETTINGPO" in text_path:
        doc = minidom.parse(text_path)
        doc_root = doc.documentElement
        text_node = xml_parse.get_xmlnode(doc_root, "TEXT")[0]
        text = xml_parse.get_nodevalue(text_node)
        doc_id = xml_parse.get_nodevalue(
            xml_parse.get_xmlnode(doc_root,
            "DOCID")[0])
        doc_type = xml_parse.get_nodevalue(
            xml_parse.get_xmlnode(doc_root,
            "DOCTYPE")[0])
        date_time = xml_parse.get_nodevalue(
            xml_parse.get_xmlnode(doc_root,
            "DATETIME")[0])
        try:
            head_line = xml_parse.get_nodevalue(
                xml_parse.get_xmlnode(doc_root,
                                      "HEADLINE")[0]).replace("\n", " ")
            sub = len(doc_id) + len(doc_type) + len(date_time) + len(head_line) + 8
        except:
            sub = len(doc_id) + len(doc_type) + len(date_time) + 6
        tokens, anchors = read_document(text, sub,event_start, 
                                        event_end, event_ident, event_map, event)
 
        return [tokens], [anchors]

    elif "wl" in text_path:
        doc = minidom.parse(text_path)
        doc_root = doc.documentElement
        post_node = xml_parse.get_xmlnode(doc_root, "POST")[0]
        text = xml_parse.get_nodevalue(post_node, 4)
        doc_id = xml_parse.get_nodevalue(
            xml_parse.get_xmlnode(doc_root,
            "DOCID")[0])
        doc_type = xml_parse.get_nodevalue(
            xml_parse.get_xmlnode(doc_root,
            "DOCTYPE")[0])
        date_time = xml_parse.get_nodevalue(
            xml_parse.get_xmlnode(doc_root,
            "DATETIME")[0])
        poster = xml_parse.get_nodevalue(
            xml_parse.get_xmlnode(doc_root,
            "POSTER")[0])
        post_date = xml_parse.get_nodevalue(
            xml_parse.get_xmlnode(doc_root,
            "POSTDATE")[0])
        sub = len(doc_id) + len(doc_type) + len(date_time) + len(poster) + len(post_date)
        try:
            head_line = xml_parse.get_nodevalue(
                xml_parse.get_xmlnode(doc_root,
                                      "HEADLINE")[0]).replace("\n", " ")
            sub = sub + len(head_line) + 10
        except:
            sub = sub + 8
        tokens, anchors = read_document(text, sub,event_start, 
                                        event_end, event_ident, event_map, event)
 
        return [tokens], [anchors]

def read_document(doc, sub, event_start, event_end, event_ident, event_map, event):
    regions = []
    tokens = []
    anchors = []
    check= 0
    offset = 0
    current = 0
    for i in range(len(doc)):
        if i+sub in event_start:
            inc = 0
            new = clean_str(doc[current:i])
            regions.append(new)
            tokens += new.split()
            check = 1
            anchors += [0 for _ in range(len(new.split()))]
            inc = 0
            current = i
            ent = event_start[i+sub]
            event[ent][2] += offset + inc
        if i+sub in event_end:
            ent = event_end[i+sub]
            event[ent][3] += offset
            new = clean_str(doc[event[ent][2]-sub : event[ent][3]-sub])
            assert new.replace(" ", "") == event[ent][4] or new == event[ent][4] or new.replace(" ","_")\
            ,"loi text: " + new + " ," + event[ent][4] +" " + str(event[ent][2]-sub) + " " + str(event[ent][3]-sub)
            regions.append(new)
            tokens += [new.replace(" ", "")]
            anchors += [EVENT_MAP[event[ent][1]]]
            offset += inc
            current = i 
    new = clean_str(doc[current : ])
    regions.append(new)
    tokens += new.split()
    anchors += [0 for _ in range(len(new.split()))]
    doc = "".join(regions)
    assert len(tokens) == len(anchors),"sai cmnr"
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

def read_corpus(folder_path):
    count = 0
    file_list = encode_corpus(folder_path)
    tokens, anchors = [], []
    for (file, path) in file_list:
        file_path = os.path.join(folder_path, path, file)
        print(file_path)
        tok, anc = read_file(file_path + ".apf.xml", file_path + ".sgm")
        count += 1
        tokens += tok
        anchors += anc
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
    #read_file(test+".apf.xml", test+".sgm", event_type = [])
    tokens, anchors  = read_corpus(
        "../ace_2005_td_v7/data/English/bn")
    t, a = read_corpus(
        "../ace_2005_td_v7/data/English/nw")
    tokens += t
    anchors += a
    pickle.dump(tokens, open("../tokens1.bin","wb"))
    pickle.dump(anchors, open("../anchors1.bin", "wb"))
    
    '''
    t, a = read_corpus(
        "../ace_2005_td_v7/data/English/bc")
    tokens = t
    anchors = a
    pickle.dump(tokens, open("tokens2.bin","wb"))
    pickle.dump(anchors, open("anchors2.bin", "wb"))
    '''

    
    t, a = read_corpus(
        "../ace_2005_td_v7/data/English/cts")
    tokens = t
    anchors = a
    pickle.dump(tokens, open("tokens2.bin","wb"))
    pickle.dump(anchors, open("anchors2.bin", "wb"))
    '''
    t, a = read_corpus(
        "../ace_2005_td_v7/data/English/wl")
    tokens = t
    anchors = a
    pickle.dump(tokens, open("../tokens2.bin","wb"))
    pickle.dump(anchors, open("../anchors2.bin", "wb"))
    '''


