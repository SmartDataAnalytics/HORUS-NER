import ntpath

from horus.core.config import HorusConfig

config = HorusConfig()


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

html_escape_table = {
        "&": "&amp;",
        '"': "&quot;",
        "'": "&apos;",
        ">": "&gt;",
        "<": "&lt;",
    }

def html_escape(text):
    return "".join(html_escape_table.get(c, c) for c in text)

def get_ner_mapping2_loop(x, y, ix, term):

    index_token = -1

    try:
        # http://stackoverflow.com/questions/32185072/nltk-word-tokenize-behaviour-for-double-quotation-marks-is-confusing
        if config.models_pos_tag_lib == 1:
            if term == '``' or term == '\'\'':
                term = u'"'

        for i in range(len(x)):
            x[i] = x[i].replace('``', u'"')
            # x[i] = x[i].replace('\'\'', u'"')

        xpos = (x[ix + 1] if ix + 1 < len(x) else '')
        xpre = (x[ix - 1] if ix > 0 else '')
        xpos2 = (x[ix + 1] + x[ix + 2] if ix + 2 < len(x) else '')
        xpre2 = (x[ix - 2] + x[ix - 1] if ix > 1 else '')

        print '================================================================'
        print 'x = %s' % (x)
        print 'y = %s' % (y)

        print 'ix = %s, term = %s' % (ix, term)
        print 'xpre2 = %s, xpre = %s, xpos = %s, xpos2 = %s' % (xpre2, xpre, xpos, xpos2)

        q = True
        # optimization trick
        start = 0
        # if i >= 14:
        #    start = i - 14
        # elif i >= 13:
        #    start = i - 13
        # elif i >= 12:
        #    start = i - 12
        # tries to get a single not aligned token
        for z in range(start, len(y)):
            try:

                ypos = (y[z + 1] if z + 1 < len(y) else '')
                ypre = (y[z - 1] if z > 0 else '')
                ypos2 = (y[z + 1] + y[z + 2] if z + 2 < len(y) else '')
                ypre2 = (y[z - 2] + y[z - 1] if z > 1 else '')

                print '----------------------------------------------------------'
                print 'ypre2 = %s, ypre = %s, ypos = %s, ypos2 = %s' % (ypre2, ypre, ypos, ypos2)
                print 'z: y[z] = %s [%s]' % (z, y[z])

                fine1 = (xpos == ypos2)
                fine2 = (xpos == ypos)
                fine3 = (xpos2 == ypos2)
                fine4 = (xpos2 == ypos)

                fine5 = (xpre == ypre2)
                fine6 = (xpre == ypre)
                fine7 = (xpre2 == ypre2)
                fine8 = (xpre2 == ypre)

                p = '_'
                if ix + 1 < len(x):
                    p = (term + x[ix + 1])
                if (y[z] == term or y[z] == p) and (
                        fine1 or fine2 or fine3 or fine4 or fine5 or fine6 or fine7 or fine8):
                    #  ok, is the correct one by value and position
                    index_token = y.index(y[z])
                    q = False
                    break
            except Exception:
                continue
        # start to merge stuff and try to locate it
        merged = ''
        print '-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-'
        ntimes = len(x) - start
        print 'ntimes = %s' % (ntimes)
        while q is True:
            for slide in range(ntimes):
                print 'slide = %s' % (slide)
                merged = ''
                if q is False:
                    break
                for m in range(start, len(x)):  # start, len(x)
                    xm = x[m].replace(u'``', u'"').replace('\'\'', u'"')
                    merged = merged + xm
                    print 'm = %s, xm = %s, merged = %s' % (m, xm, merged)
                    try:
                        index_token = y.index(merged)
                        af = (x[ix + 1] if ix + 1 < len(x) else '')
                        bf = (x[ix - 1] if ix > 0 else '')

                        af = af.replace(u'``', u'"')  # .replace('\'\'', u'"')
                        bf = bf.replace(u'``', u'"')

                        print 'af = %s, bf = %s' % (af, bf)

                        if term in merged and (
                                af in merged or bf in merged):  # if it is merged, at least 2 MUST be included
                            q = False
                            break
                    except Exception:
                        continue
                start += 1
            if q is True:
                return None

    except Exception as e:
        print(':: error on get ner: %s' % e)
        exit(-1)

    return index_token

def get_ner_mapping2(y, x, t, i):
    if i + 1 < len(y):
        if y[i] == t:
            return i
    index_token = get_ner_mapping2_loop(x, y, i, t)
    if index_token is None:
        # means that we looked over all combinations in x that could exist in y
        # if we enter here, means that it is not possible and thus, the given token t has not been
        # tokenized in x, but in y yes! try other way around...
        index_token = get_ner_mapping2_loop(y, x, i, t)
        if index_token is None:
            print 'it should never ever happen!!! maiden maiden!'
            exit(-1)
    return index_token

def get_ner_mapping(listy, listx, termx, itermx):
    index_ner_y = -1
    try:
        # lists are aligned
        if listy[itermx] == listx[itermx]:
            return itermx
        else:
            # simplest solution
            for itermy in range(len(listy) - 1):
                if listy[itermy] == termx and (listy[itermy - 1] == listx[itermx - 1]
                                               or listy[itermy + 1] == listx[itermx + 1]):
                    index_ner_y = itermy
                    break
            if index_ner_y != -1:
                return index_ner_y
            else:
                try:
                    # from this point on, it' more tricky, so let' start
                    # if t isn't there, automatically, t+1 will not be there also!
                    # Thus, try to catch t+2, i.e.,
                    # we gonna have t that has been tokenized in two parts
                    # (excepting if it' about the last item)
                    if itermx + 2 <= len(listx):
                        next_2_term = listx[itermx + 2]
                        listy.index(next_2_term)  # dummy var - this will raise an error in case of problems
                    # if worked, then merging the 2 before, will make it work
                    term_merged = termx + listx[itermx + 1]
                    index_token = listy.index(term_merged)
                    index_ner_y = index_token
                except:
                    # checking if current is last
                    try:
                        term_merged = listx[itermx - 1] + termx
                        index_token = listy.index(term_merged)
                        index_ner_y = index_token
                    except:  # try now i + i+1 + i+2!
                        try:
                            term_merged = termx + listx[itermx + 1] + listx[itermx + 2]
                            index_token = listy.index(term_merged)
                            index_ner_y = index_token
                        except:
                            # checking if current is last
                            try:
                                term_merged = listx[itermx - 2] + listx[itermx - 1] + termx
                                index_token = listy.index(term_merged)
                                index_ner_y = index_token
                            except:
                                print 'maiden maiden...!!!!!'
                                print termx, itermx
                                exit(-1)

    except Exception as error:
        print(':: error on get ner: %s' % error)

    return index_ner_y

def get_ner_mapping_slice(y, x, ix):

    try:
        for i in range(len(x)):
            x[i] = x[i].replace('``', u'"')
            # x[i] = x[i].replace("''", u'"')

        ##################################################
        # cases (|count(left)| + x + |count(right)| = 7)
        #################################################
        # d
        term = x[ix]
        # d + 6
        merged_aft_7 = x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] + x[ix + 4] + x[ix + 5] + x[ix + 6] + x[ix + 7] \
            if ix + 7 < len(x) else ''
        merged_aft_6 = x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] + x[ix + 4] + x[ix + 5] + x[ix + 6] \
            if ix + 6 < len(x) else ''
        merged_aft_5 = x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] + x[ix + 4] + x[ix + 5] if ix + 5 < len(x) else ''
        merged_aft_4 = x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] + x[ix + 4] if ix + 4 < len(x) else ''
        merged_aft_3 = x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] if ix + 3 < len(x) else ''
        merged_aft_2 = x[ix] + x[ix + 1] + x[ix + 2] if ix + 2 < len(x) else ''
        merged_aft_1 = x[ix] + x[ix + 1] if ix + 1 < len(x) else ''

        # d - 7
        merged_bef_7 = x[ix - 7] + x[ix - 6] + x[ix - 5] + x[ix - 4] + x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] \
            if ix >= 7 else ''
        merged_bef_6 = x[ix - 6] + x[ix - 5] + x[ix - 4] + x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] \
            if ix >= 6 else ''
        merged_bef_5 = x[ix - 5] + x[ix - 4] + x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] \
            if ix >= 5 else ''
        merged_bef_4 = x[ix - 4] + x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] \
            if ix >= 4 else ''
        merged_bef_3 = x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] \
            if ix >= 3 else ''
        merged_bef_2 = x[ix - 2] + x[ix - 1] + x[ix] \
            if ix >= 2 else ''
        merged_bef_1 = x[ix - 1] + x[ix] \
            if ix >= 1 else ''

        # -1 d +1
        merged_bef_1_aft_1 = x[ix - 1] + x[ix] + x[ix + 1] \
            if (ix + 1 < len(x) and ix >= 1) else ''
        # -2 d +2
        merged_bef_2_aft_2 = x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] \
            if (ix + 2 < len(x) and ix >= 2) else ''
        # -3 d +3
        merged_bef_3_aft_3 = x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] \
            if (ix + 3 < len(x) and ix >= 3) else ''

        # -1 d +2..5
        merged_bef_1_aft_2 = x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] \
            if (ix + 2 < len(x) and ix >= 1) else ''
        merged_bef_1_aft_3 = x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] \
            if (ix + 3 < len(x) and ix >= 1) else ''
        merged_bef_1_aft_4 = x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] + x[ix + 4] \
            if (ix + 4 < len(x) and ix >= 1) else ''
        merged_bef_1_aft_5 = x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] + x[ix + 4] + x[ix + 5] \
            if (ix + 5 < len(x) and ix >= 1) else ''
        merged_bef_1_aft_6 = x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] + x[ix + 4] + x[ix + 5] + x[
            ix + 6] \
            if (ix + 6 < len(x) and ix >= 1) else ''

        # -2..5 d +1
        merged_bef_2_aft_1 = x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] \
            if (ix + 1 < len(x) and ix >= 2) else ''
        merged_bef_3_aft_1 = x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] \
            if (ix + 1 < len(x) and ix >= 3) else ''
        merged_bef_4_aft_1 = x[ix - 4] + x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] \
            if (ix + 1 < len(x) and ix >= 4) else ''
        merged_bef_5_aft_1 = x[ix - 5] + x[ix - 4] + x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] \
            if (ix + 1 < len(x) and ix >= 5) else ''
        merged_bef_6_aft_1 = x[ix - 6] + x[ix - 5] + x[ix - 4] + x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] + x[
            ix + 1] \
            if (ix + 1 < len(x) and ix >= 6) else ''
        merged_bef_5_aft_2 = x[ix - 5] + x[ix - 4] + x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] + x[
            ix + 2] \
            if (ix + 2 < len(x) and ix >= 5) else ''

        # -2 d +3..5
        merged_bef_2_aft_3 = x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] \
            if (ix + 3 < len(x) and ix >= 2) else ''
        merged_bef_2_aft_4 = x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] + x[ix + 4] \
            if (ix + 4 < len(x) and ix >= 2) else ''
        merged_bef_2_aft_5 = x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] + x[ix + 4] + x[
            ix + 5] \
            if (ix + 5 < len(x) and ix >= 2) else ''

        # -3..4 d +2
        merged_bef_3_aft_2 = x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] \
            if (ix + 2 < len(x) and ix >= 3) else ''
        merged_bef_3_aft_4 = x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] + x[ix + 3] + x[
            ix + 4] \
            if (ix + 4 < len(x) and ix >= 3) else ''
        merged_bef_4_aft_2 = x[ix - 4] + x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] \
            if (ix + 2 < len(x) and ix >= 4) else ''
        merged_bef_4_aft_3 = x[ix - 4] + x[ix - 3] + x[ix - 2] + x[ix - 1] + x[ix] + x[ix + 1] + x[ix + 2] \
                             + x[ix + 3] if (ix + 3 < len(x) and ix >= 4) else ''

        seq = [[term, -1, 1],
               [merged_aft_1, -1, 2],
               [merged_aft_2, -1, 3],
               [merged_aft_3, -1, 4],
               [merged_aft_4, -1, 5],
               [merged_aft_5, -1, 6],
               [merged_aft_6, -1, 7],
               [merged_aft_7, -1, 8],
               [merged_bef_1, -2, 1],
               [merged_bef_2, -3, 1],
               [merged_bef_3, -4, 1],
               [merged_bef_4, -5, 1],
               [merged_bef_5, -6, 1],
               [merged_bef_6, -7, 1],
               [merged_bef_7, -8, 1],
               [merged_bef_1_aft_1, -2, 2],
               [merged_bef_2_aft_2, -3, 3],
               [merged_bef_3_aft_3, -4, 4],
               [merged_bef_1_aft_2, -2, 3],
               [merged_bef_1_aft_3, -2, 4],
               [merged_bef_1_aft_4, -2, 5],
               [merged_bef_1_aft_5, -2, 6],
               [merged_bef_1_aft_6, -2, 7],
               [merged_bef_2_aft_1, -3, 2],
               [merged_bef_3_aft_1, -4, 2],
               [merged_bef_4_aft_1, -5, 2],
               [merged_bef_5_aft_1, -6, 2],
               [merged_bef_2_aft_3, -3, 4],
               [merged_bef_2_aft_4, -3, 5],
               [merged_bef_2_aft_5, -3, 6],
               [merged_bef_3_aft_2, -4, 3],
               [merged_bef_4_aft_2, -5, 3],
               [merged_bef_4_aft_3, -5, 4],
               [merged_bef_6_aft_1, -7, 2],
               [merged_bef_5_aft_2, -6, 3],
               [merged_bef_3_aft_4, -4, 5]]

        for s in seq:
            xbefore1 = x[ix + s[1]] if (ix + s[1]) >= 0 else ''

            xbefore2 = x[ix + s[1] - 1] + x[ix + s[1]] \
                if (ix + s[1] - 1) >= 0 else ''

            xbefore3 = x[ix + s[1] - 2] + x[ix + s[1] - 1] + x[ix + s[1]] \
                if (ix + s[1] - 2) >= 0 else ''

            xbefore4 = x[ix + s[1] - 3] + x[ix + s[1] - 2] + x[ix + s[1] - 1] + x[ix + s[1]] \
                if (ix + s[1] - 3) >= 0 else ''

            xbefore5 = x[ix + s[1] - 4] + x[ix + s[1] - 3] + x[ix + s[1] - 2] + x[ix + s[1] - 1] + x[ix + s[1]] \
                if (ix + s[1] - 4) >= 0 else ''

            xafter4 = x[ix + s[2]] + x[ix + s[2] + 1] + x[ix + s[2] + 2] + x[ix + s[2] + 3] \
                if (ix + s[2] + 3) < len(x) else ''

            xafter3 = x[ix + s[2]] + x[ix + s[2] + 1] + x[ix + s[2] + 2] \
                if (ix + s[2] + 2) < len(x) else ''

            xafter2 = x[ix + s[2]] + x[ix + s[2] + 1] \
                if (ix + s[2] + 1) < len(x) else ''

            xafter1 = x[ix + s[2]] \
                if (ix + s[2]) < len(x) else ''

            for iy in range(len(y)):
                ybefore = y[iy - 1] if iy > 0 else ''
                yafter = y[iy + 1] if iy + 1 < len(y) else ''
                print '    ybefore: %s, y: %s, yafter: %s' % (ybefore, y[iy], yafter)
                if (y[iy] == s[0] or y[iy] == s[0].replace(u'"', u"''")) and (ybefore == xbefore1 or
                                                                              ybefore == xbefore2 or
                                                                              ybefore == xbefore3 or
                                                                              ybefore == xbefore4 or
                                                                              ybefore == xbefore5 or
                                                                              yafter == xafter1 or
                                                                              yafter == xafter2 or
                                                                              yafter == xafter3 or
                                                                              yafter == xafter4):
                    return iy

        print 'index not found'
        exit(-1)
    except Exception as error:
        exit(-1)

def remove_non_ascii(t):
    import string
    printable = set(string.printable)
    temp = filter(lambda x: x in printable, t)
    return temp
    # return "".join(i for i in temp if ord(i) < 128)

def convert_unicode(s):
    # u'abc'.encode('utf-8') -> unicode to str
    # 'abc'.decode('utf-8') -> str to unicode
    if isinstance(s, str):
        return s.decode('utf8')  # unicode(s, 'utf8 )
    elif isinstance(s, unicode):
        return s
    else:
        raise Exception("that's not a string!")