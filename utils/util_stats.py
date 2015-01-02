#!/usr/bin/env python


if __name__ == '__main__':
    import os
    import struct
    import time
    import sys
    import numpy as np
    import matplotlib.pyplot as plt


    def hist(dictionary, val):
        if val in dictionary:
            dictionary[val] += 1
        else:
            dictionary[val] = 1

    if len(sys.argv) == 1:
        print "Usage: python path_to_folder [output_filename] [-v]"

    if len(sys.argv) >= 2:
        path = sys.argv[1].strip().strip("/")

        if not os.path.isdir(path):
            print "[ ERROR ] Specified path does not exist"
            sys.exit(0)

    files = []
    for filename in os.listdir(path):
        if filename.endswith(".txt") and filename != "config.txt":
            files.append(path + "/" + filename)

    print "Number of files to stat:", len(files)


    alphas = {}
    ps = {}
    qs = {}
    rs = {}
    ss = {}
    taos = {}

    for i in range(len(files)):
    # for i in []:
        treefilename = files[i]
        treefile = open(treefilename)

        initial = treefile.readline().strip()
        initial = initial.split(" ")

        print "Processing:", treefilename, "   [%7.2f %s ]" %(100 * float(i) / len(files), "%")

        for line in treefile:
            line = line.strip()

            if line == "":
                break

            line = line.split(" ")
            index = int(line[0])
            line = map(int, line[3:])

            if line != [0] * 6:
                hist(ps, line[0])
                hist(qs, line[1])
                hist(rs, line[2])
                hist(ss, line[3])
                hist(alphas, line[4])
                hist(taos, line[5])

    dicts = [(alphas, "alpha", "y"), (taos, "tao", "g"), (ps, "p", "b"), (qs, "q", "c"), (rs, "r", "r"), (ss, "s", "m")]

    opacity = 0.4
    bar_width = 0.9

    fig, ax = plt.subplots(3,2)
    # plt.tight_layout()

    for i in range(len(dicts)):
        source = dicts[i][0]
        title = dicts[i][1]
        color = dicts[i][2]

        temp_index = sorted(source.keys())
        temp = [source[key] for key in temp_index]

        y = i % 2
        x = i / 2
        ax[x,y].bar(temp_index, temp, bar_width,
                         alpha=opacity,
                         color=color)
        ax[x,y].set_title(title)
        ax[x,y].set_xlim([min(temp_index),max(temp_index)])

    plt.show()
