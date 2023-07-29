def idlist_to_listid(idlist):
    lst = []
    while idlist:
        lst.append(idlist.ID())
        idlist = idlist.idlist()
    return lst

def explist_to_listexp(explist):
    lst = []
    while explist:
        lst.append(explist.exp())
        explist = explist.explist()
    return lst

def arglist_to_listarg(arglist):
    lst = []
    while arglist:
        lst.append(arglist.argument())
        arglist = arglist.arglist()
    return lst


def goplist_to_listgop(goplist):
    lst = []
    while goplist:
        lst.append(goplist.gop())
        goplist = goplist.goplist()
    return lst
