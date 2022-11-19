import numpy as np
from struct import unpack
import jdata as jd
from collections import OrderedDict
import sys
import os


def loadmc2(path, dimension):

    f = open(path, 'rb')
    data = f.read()
    data = unpack('%df' % (len(data) / 4), data)
    data = np.asarray(data).reshape(dimension, order='F')
    return data


def loadmch(fname, format='f', endian='ieee-le', datadict=False):
    def fread(fileid, N, Type):
        if Type == 'c' or Type == 'b' or Type == 'B' or Type == '?':
            Nb = 1
        elif Type == 'h' or Type == 'H':
            Nb = 2
        elif Type == 'i' or Type == 'I' or Type == 'l' or Type == 'L' or Type == 'f':
            Nb = 4
        elif Type == 'q' or Type == 'Q' or Type == 'd':
            Nb = 8
        else:
            raise Exception("Type unknow")

        if N == 1:
            return unpack(Type, fileid.read(Nb))[0]
        else:
            return unpack(str(N) + Type, fileid.read(N * Nb))

    try:
        fid = open(fname, 'rb')
    except:
        raise Exception("Could no open the given file name " + fname)

    data = []
    header = []
    photon_seed = []

    while True:

        magicheader = fid.read(4)  # a char is 1 Bytes

        if not magicheader:
            break
        elif magicheader != b'MCXH':
            fid.close()
            raise Exception("It might not be a mch file!")

        version = fread(fid, 1, 'I')

        assert version == 1, "version higher than 1 is not supported"

        maxmedia = fread(fid, 1, 'I')
        detnum = fread(fid, 1, 'I')
        colcount = fread(fid, 1, 'I')
        totalphoton = fread(fid, 1, 'I')
        detected = fread(fid, 1, 'I')
        savedphoton = fread(fid, 1, 'I')
        unitmm = fread(fid, 1, 'f')
        seedbyte = fread(fid, 1, 'I')
        normalizer = fread(fid, 1, 'f')
        respin = fread(fid, 1, 'i')
        srcnum = fread(fid, 1, 'I')
        savedetflag = fread(fid, 1, 'I')
        junk = fread(fid, 2, 'i')

        detflag = np.asarray(list(bin(savedetflag & (2 ** 8 - 1))[2:]), 'int')
        if endian == 'ieee-le': detflag = detflag[::-1]  # flip detflag left to right
        datalen = np.asarray([1, maxmedia, maxmedia, maxmedia, 3, 3, 1])
        datlen = detflag * datalen[0:len(detflag)]

        dat = fread(fid, (colcount * savedphoton), format)
        dat = np.asarray(dat).reshape(savedphoton, colcount)

        if savedetflag and len(detflag) > 2 and detflag[2] > 0:
            dat[:, sum(datlen[0:2]):sum(datlen[0:3])] = dat[:, sum(datlen[0:2]):sum(datlen[0:3])] * unitmm
        elif savedetflag == 0:
            dat[:, 1 + maxmedia:(2 * maxmedia)] = dat[:, 1 + maxmedia:(2 * maxmedia)] * unitmm

        # make the data as a dictionary
        if datadict:
            if savedetflag:
                data_dic = [{} for x in range(savedphoton)]
                for photonid in range(savedphoton):
                    if len(detflag) > 0 and detflag[0] != 0: data_dic[photonid]["detid"] = dat[photonid][0]
                    if len(detflag) > 1 and detflag[1] != 0: data_dic[photonid]["nscat"] = dat[photonid][
                                                                                           datlen[0]:1 + datlen[1]]
                    if len(detflag) > 2 and detflag[2] != 0: data_dic[photonid]["ppath"] = dat[photonid][
                                                                                           sum(datlen[0:2]):sum(
                                                                                               datlen[0:3])]
                    if len(detflag) > 3 and detflag[3] != 0: data_dic[photonid]["mom"] = dat[photonid][
                                                                                         sum(datlen[0:3]):sum(
                                                                                             datlen[0:4])]
                    if len(detflag) > 4 and detflag[4] != 0: data_dic[photonid]["p"] = dat[photonid][
                                                                                       sum(datlen[0:4]):sum(
                                                                                           datlen[0:5])]
                    if len(detflag) > 5 and detflag[5] != 0: data_dic[photonid]["v"] = dat[photonid][
                                                                                       sum(datlen[0:5]):sum(
                                                                                           datlen[0:6])]
                    if len(detflag) > 6 and detflag[6] != 0: data_dic[photonid]["w0"] = dat[photonid][-1]

            elif savedetflag == 0:
                data_dic = [{"detid": photon[0],
                             "nscat": photon[1:1 + maxmedia],
                             "ppath": photon[1 + maxmedia:1 + 2 * maxmedia],
                             "mom": photon[1 + 2 * maxmedia:1 + 3 * maxmedia],
                             "p": photon[-7:-4:1], "v": photon[-4:-1:1],
                             "w0": photon[-1]} for photon in dat]

            del dat
            dat = np.asarray(data_dic)

        data.append(dat)

        # if "save photon seed" is True
        if seedbyte > 0:
            # seeds = unpack('%dB' % (savedphoton*seedbyte), fid.read(savedphoton*seedbyte))
            seeds = fread(fid, (savedphoton * seedbyte), 'B')
            photon_seed.append(np.asarray(seeds).reshape((seedbyte, savedphoton), order='F'))

        if respin > 1: totalphoton *= respin

        header = {'version': version,
                  'medianum': maxmedia,
                  'detnum': detnum,
                  'recordnum': colcount,
                  'totalphoton': totalphoton,
                  'detectedphoton': detected,
                  'savedphoton': savedphoton,
                  'lengthunit': unitmm,
                  'seedbyte': seedbyte,
                  'normalizer': normalizer,
                  'respin': respin,
                  'srcnum': srcnum,
                  'savedetflag': savedetflag}

    fid.close()

    data = np.asarray(data).squeeze()

    if seedbyte > 0:
        photon_seed = np.asarray(photon_seed).transpose((0, 2, 1)).squeeze()

    return data, header, photon_seed


def mcxtry(shuru_image):
    datadict = False
    yahau = np.reshape(shuru_image, [256, 256])
    yahau = np.expand_dims(yahau, -1).repeat(256, axis=-1)
    # yahau = shuru_image
    # yahau = np.reshape(shuru_image, [256, 256, 256])
    cfg = OrderedDict()
    cfg = {
        'Session': {
            'ID': 'absorrand',
            'Photons': 1e8
        },
        'Forward': {
            'T0': 0,
            'T1': 5e-09,
            'Dt': 5e-09
        },
        'Domain': {
            'MediaFormat': 'byte',
            'LengthUnit': 0.1,

            'Media': [{"mua": 0,
                       "mus": 0,
                       "g": 1,
                       "n": 1}
                      ],

            'Dim': [256, 256, 256],
            'OriginType': 1
        },
        'Optode': {
            'Source': {
                "Type": "slit", "Pos": [0, 0, 127], "Dir": [1, 0, 0, 0], "Param1": [0, 256, 0, 0],
                "Param2": [0, 0, 0, 0],
            },
            'Detector': []
        },
        'Shapes': yahau
    }

    for i in range(1, 101):
        ua_true = i / 1000
        float('%.3f' % ua_true)
        if i == 1:
            yaha = {"mua": 0.0, "mus": 0, "g": 1, "n": 1}  # what a delightful bug !
        else:
            yaha = {"mua": ua_true, "mus": 10, "g": 0.9, "n": 1.37}
        cfg["Domain"]["Media"].append(yaha)

    guangyuan1 = {"Type": "slit", "Pos": [0, 0, 127], "Dir": [1, 0, 0, 0], "Param1": [0, 256, 0, 0],
                  "Param2": [0, 0, 0, 0]}
    guangyuan2 = {"Type": "slit", "Pos": [0, 255, 127], "Dir": [0, -1, 0, 0], "Param1": [256, 0, 0, 0],
                  "Param2": [0, 0, 0, 0]}
    guangyuan3 = {"Type": "slit", "Pos": [255, 255, 127], "Dir": [-1, 0, 0, 0], "Param1": [0, -256, 0, 0],
                  "Param2": [0, 0, 0, 0]}
    guangyuan4 = {"Type": "slit", "Pos": [0, 0, 127], "Dir": [0, 1, 0, 0], "Param1": [256, 0, 0, 0],
                  "Param2": [0, 0, 0, 0]}

    cfg["Optode"]["Source"] = guangyuan1
    newdata = cfg.copy()
    aaa = jd.encode(newdata, {'compression': 'zlib', 'base64': True})
    SID = cfg["Session"]["ID"]
    jd.save(aaa, SID + '.json', indent=4)  # 同名目录下产生 absorrand.json

    # sys.path.append("F:\\wc\\原来G盘\\蒙卡\\蒙卡python\\mcx")
    # sys.path.append("D:\Aluminium\Documents\MATLAB\mcx")

    mcxbin = 'default'
    if mcxbin == 'default':
        if os.name == "posix":
            mcxbin = './mcx'
        else:
            # mcxbin = 'F:\\wc\\原来G盘\\蒙卡\\蒙卡python\\mcx\\pymcx\\pymcx\\mcx.exe'
            mcxbin = 'mcx'

    flag = "0"
    os.system(mcxbin + ' -f ' + SID + '.json ' + flag)  # 同名目录下产生 absorrand.mc2

    mch = []
    mc2 = []

    if os.path.isfile(SID + '.mch'): mch = loadmch(SID + '.mch', datadict=datadict)

    if os.path.isfile(SID + '.mc2'):
        nbstep = round((cfg["Forward"]["T1"] - cfg["Forward"]["T0"]) / cfg["Forward"]["Dt"])
        if "Dim" in cfg["Domain"] and cfg["Domain"]["Dim"] != []:
            dt = cfg["Domain"]["Dim"] + [nbstep]
        elif "Shapes" in cfg:
            for find in cfg["Shapes"]:
                if "Grid" in find:
                    dt = find["Grid"]["Size"] + [nbstep]

        mc2 = loadmc2(SID + '.mc2', dt)
        result1 = mc2

    cfg["Optode"]["Source"] = guangyuan2
    newdata = cfg.copy()
    aaa = jd.encode(newdata, {'compression': 'zlib', 'base64': True})
    jd.save(aaa, SID + '.json', indent=4)
    os.system(mcxbin + ' -f ' + SID + '.json ' + flag)

    mc3 = []

    if os.path.isfile(SID + '.mch'): mch = loadmch(SID + '.mch', datadict=datadict)

    if os.path.isfile(SID + '.mc2'):
        nbstep = round((cfg["Forward"]["T1"] - cfg["Forward"]["T0"]) / cfg["Forward"]["Dt"])
        if "Dim" in cfg["Domain"] and cfg["Domain"]["Dim"] != []:
            dt = cfg["Domain"]["Dim"] + [nbstep]
        elif "Shapes" in cfg:
            for find in cfg["Shapes"]:
                if "Grid" in find:
                    dt = find["Grid"]["Size"] + [nbstep]

        mc3 = loadmc2(SID + '.mc2', dt)
        result2 = mc3

    cfg["Optode"]["Source"] = guangyuan3
    newdata = cfg.copy()
    aaa = jd.encode(newdata, {'compression': 'zlib', 'base64': True})
    jd.save(aaa, SID + '.json', indent=4)
    os.system(mcxbin + ' -f ' + SID + '.json ' + flag)

    if os.path.isfile(SID + '.mch'): mch = loadmch(SID + '.mch', datadict=datadict)

    if os.path.isfile(SID + '.mc2'):
        nbstep = round((cfg["Forward"]["T1"] - cfg["Forward"]["T0"]) / cfg["Forward"]["Dt"])
        if "Dim" in cfg["Domain"] and cfg["Domain"]["Dim"] != []:
            dt = cfg["Domain"]["Dim"] + [nbstep]
        elif "Shapes" in cfg:
            for find in cfg["Shapes"]:
                if "Grid" in find:
                    dt = find["Grid"]["Size"] + [nbstep]

        mc2 = loadmc2(SID + '.mc2', dt)
        result3 = mc2

    cfg["Optode"]["Source"] = guangyuan4
    newdata = cfg.copy()
    aaa = jd.encode(newdata, {'compression': 'zlib', 'base64': True})
    jd.save(aaa, SID + '.json', indent=4)
    os.system(mcxbin + ' -f ' + SID + '.json ' + flag)

    if os.path.isfile(SID + '.mch'): mch = loadmch(SID + '.mch', datadict=datadict)

    if os.path.isfile(SID + '.mc2'):
        nbstep = round((cfg["Forward"]["T1"] - cfg["Forward"]["T0"]) / cfg["Forward"]["Dt"])
        if "Dim" in cfg["Domain"] and cfg["Domain"]["Dim"] != []:
            dt = cfg["Domain"]["Dim"] + [nbstep]
        elif "Shapes" in cfg:
            for find in cfg["Shapes"]:
                if "Grid" in find:
                    dt = find["Grid"]["Size"] + [nbstep]

        mc2 = loadmc2(SID + '.mc2', dt)
        result4 = mc2

    gaga = result1 + result2 + result3 + result4

    return gaga

