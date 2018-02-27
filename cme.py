import sys, string, urllib
 
GROUPS = dict([('ags','ftp://ftp.cmegroup.com/pub/settle/stlags'),
               ('rates','ftp://ftp.cmegroup.com/pub/settle/stlint'),
               ('fx','ftp://ftp.cmegroup.com/pub/settle/stlcur'),
               ('nymex','ftp://ftp.cmegroup.com/pub/settle/stlnymex'),
               ('comex','ftp://ftp.cmegroup.com/settle/stlcomex'),
               ('clearport','ftp://ftp.cmegroup.com/settle/stlcpc')])
 
# month codes
MONTHDICT = dict([(i+1, 'FGHJKMNQUVXZ'[i]) for i in range(12)])
MONTHDICT.update(dict([('FGHJKMNQUVXZ'[i], i+1) for i in range(12)]))
MONTH_NAMES = 'JAN FEB MAR APR MAY JUN JLY AUG SEP OCT NOV DEC'.split()
MONTHDICT.update(dict(zip(MONTH_NAMES, 'F G H J K M N Q U V X Z'.split())))
 
def month_code(month):
    return MONTHDICT[month]
 
def join_words(words, sep):
    """my own simple version of string.join; never deprecated"""
    result = ''
    for word in words[:-1]:
        result += word + sep
    return result + words[-1]
 
def remove_word(sentence, word, separator):
    return join_words(sentence.split(word), separator).strip()
 
def Settlements(grp):
    path = GROUPS[grp]
    u = urllib.urlopen(path)
    lines = u.readlines()
    print lines[0].strip()
    u.close()
    RES = {}
    key = None
    for l in lines:
        if len(l.strip()) == 0:
            continue
        tokens = [t for t in l.split() if len(t.strip()) > 0]
        if tokens[0] == 'TOTAL':
            continue
        if len(tokens) > 5 and tokens[1] == tokens[2] == tokens[3] == tokens[4] == '----':
            continue
        try:
            int(tokens[0]); continue
        except:
            pass

        if tokens[-1] in ['CALL', 'PUT', '***']:
            continue
        if tokens[1] == 'SYNTH':
            continue
        if tokens[-1].lower() in ['future', 'futures', 'swap', 'swp']:
            symbol = tokens[0]
            if tokens[1] == 'Mini-sized':
                tokens[1] = 'Mini'
            key = (symbol, string.join(tokens[1:], ' ').upper().strip())
            RES[key] = []
            if symbol == 'ED':
                count = 24
            else:
                count = 3
            continue
        if key is not None:
            x = dict(zip('EXP OPEN HI LO LAST SETT CHNG'.split(), tokens[:7]))
            try:
                x['EXP'] = month_code(x['EXP'][:3]) + x['EXP'][-1]
            except:
                key = None
                continue
            RES[key].append(x)
            count -= 1
            if count == 0:
                key = None
            continue
    return RES


def DisplaySettlements(grp):
    result = Settlements(grp)
    keys = result.keys()
    keys.sort()
    output = []
    for (symbol, description) in keys:
        if len(result[(symbol, description)]) == 0:
            continue
        if description == 'MINNEAPOLIS HARD RED SPRING WHEAT FUTURES':
            output.append('MN SPRING WHEAT FUTURES')
        elif description == 'MINNEAPOLIS SOFT RED WINTER WHEAT INDEX FUTURES':
            output.append('MN WINTER WHEAT IDX FUTURES')
        elif description == 'KANSAS CITY WHEAT FUTURES':
            output.append('KC WHEAT FUTURES')
        else:
            d = remove_word(description, 'CME', '')
            output.append(d)
        for d in result[(symbol, description)]:
            line = string.join(['     ', (symbol + d['EXP']).ljust(10),
                                d['SETT'].ljust(8), d['CHNG'].ljust(8)])
            output.append(line)
    return output
 
#if __name__ == '__main__':
 
    #op = DisplaySettlements('ags')
#    op = DisplaySettlements('comex')
     
#    print string.join(op, '\n')
