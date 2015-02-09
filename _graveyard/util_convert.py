#!/usr/bin/env python

'''
	Converts Gall et. al. tree .txt files into the Barinova et. al. tree binary file
'''

import os
import struct
import time
import sys

class Tree(object):
	def __init__(self, i, val, d, data = "[ NONE ]"):
		self.data = data
		self.index = i
		self.val = val
		self.d = d
		self.out = None
		self.leaf = None
		self.left = None
		self.right = None

	def depth(self):
		if self.left == None and self.right == None:
			return 1
		if self.left == None:
			return self.right.depth() + 1
		if self.right == None:
			return self.left.depth() + 1
		
		return max(self.left.depth(), self.right.depth()) + 1

	def members(self):
		if self.left == None:
			left = 0
		else:
			left = self.left.members()

		if self.right == None:
			right = 0
		else:
			right = self.right.members()
		
		return 1 + left + right

	def leaves(self):
		temp = [0] * 7
		if self.left == None and self.right == None:
			return 1
		if self.left == None:
			return self.right.leaves()
		elif self.right == None:
			return self.left.leaves()
		elif compare(self.left.data, temp) or compare(self.right.data, temp):
			return 1
		else:
			return self.left.leaves() + self.right.leaves()

	def add(self, i, val, d, data, traversal):
		if traversal == "L":
			self.left = Tree(i, val, d, data)
		elif traversal == "R":
			self.right = Tree(i, val, d, data)
		elif traversal[0] == "L":
			self.left.add(i, val, d, data, traversal[1:])
		elif traversal[0] == "R":
			self.right.add(i, val, d, data, traversal[1:])

	def prepare(self, c = 0):
		self.out = c
		c += 1
		
		temp = [0] * 7

		if self.left != None and self.data[0] == -1:
			c = self.left.prepare(c)

		if self.right != None and self.data[0] == -1:
			c = self.right.prepare(c)

		return c

	def output(self):
		global usedcount, verbose

		if self.left == None:
			left = None
		else:
			left = self.left.out

		if self.right == None:
			right = None
		else:
			right = self.right.out

		if left == None and right == None:
			# if verbose:
			# 	print "Leaf Assignment Verify:", usedcount, self.data[0]
			# 	print "     ", left, right

			# Verify leaves being assigned correctly
			self.leaf = leaves[usedcount]
			assert usedcount == self.data[0]
			
			usedcount += 1
			ratio = (1 if self.leaf[0] == 0 else int(round((len(self.leaf[1]) / 2) / self.leaf[0])))

			for j in range(len(self.leaf[1])):
				self.leaf[1][j] = self.leaf[1][j] * -1 + 8

			if verbose:
				print self.out, self.d, 1
				print "\t", ratio, len(self.leaf[1]) / 2, self.leaf[1]

			# generator(self.out)
			generator(self.d)
			generator(1, "?")
			generator(ratio) # Ratio
			generator(len(self.leaf[1]) / 2)
			generator(self.leaf[1])
		else:
			temp_vector = [self.data[1], self.data[2], self.data[1], self.data[2], self.data[3], self.data[4], self.data[3], self.data[4]]
			if verbose:
				print self.out, self.d, 0, self.data[5], temp_vector, self.data[6] - 1, right, left
			# generator(self.out)
			generator(self.d)
			generator(0, "?")
			generator(self.data[5]) # Channel
			generator(temp_vector) #Vector
			generator(self.data[6] - 1) # Tau
			generator(right)
			generator(left)
		
		temp = [0] * 7
		
		if self.left != None and self.data[0] == -1:
			self.left.output()

		if self.right != None and self.data[0] == -1:
			self.right.output()

def place(n):
	n += 1

	level = 0

	while (2 ** level) <= n:
		level += 1

	level -= 1
	space = n % (2 ** level)

	retVal = ""
	
	while level > 0:
		level -= 1

		if space % 2 == 1:
			retVal += "R"
		else:
			retVal += "L"

		space /= 2

	return retVal[::-1]

def compare(list1, list2):
	for i in range(len(min(list1, list2))):
		if list1[i] != list2[i]:
			return False

	return True

def generator(value, fmt="i"):
	global outputfile

	if type(value) == list:
		for val in value:
			outputfile.write(struct.pack(fmt, int(val)))
	else:
		outputfile.write(struct.pack(fmt, int(value)))

verbose = "-v" in sys.argv
default_output = "output.dat"

if len(sys.argv) == 1:
	print "Usage: python path_to_folder [output_filename] [-v]"

if len(sys.argv) >= 2:
	path = sys.argv[1].strip().strip("/")

	if not os.path.isdir(path):
		print "[ ERROR ] Specified path does not exist"
		sys.exit(0)

if len(sys.argv) >= 3:
	default_output = sys.argv[2]

outputfile = open(path + "/" + default_output, "wb")

files = []
for filename in os.listdir(path):
    if filename.endswith(".txt") and filename != "config.txt":
        files.append(path + "/" + filename)

if verbose:
	print "Number of files to convert:", len(files)

generator(len(files))

for i in range(len(files)):
	treefilename = files[i]
	treefile = open(treefilename)

	initial = treefile.readline().strip()
	initial = initial.split(" ")
	depth_start = int(initial[0])
	leaves_start = int(initial[1])

	if verbose:
		print "\n----------------------------------------"

	print "Processing:", treefilename, "   [%7.2f %s ]" %(100 * float(i) / len(files), "%")
	
	if verbose:
		print "Depth:", depth_start
		print "Leaves:", leaves_start
		print "----------"

	line = treefile.readline()
	line = line.strip()
	line = line.split(" ")
	line = map(int, line[2:])
	tree = Tree(-1, 0, 0, line)

	val = 1
	for line in treefile:
		line = line.strip()

		if line == "":
			break

		line = line.split(" ")
		index = int(line[0])
		line = map(int, line[2:])

		travers = place(val)
		tree.add(index, val, len(travers), line, travers)
		val += 1

	leaves = []
	for line in treefile:
		line = line.strip()
		line = line.split(" ")
		ratio = float(line[1])
		line = map(int, line[3:])
		leaves.append([ratio, line])

	usedcount = 0
	nodes = tree.prepare()
	if verbose:
		print nodes
	generator(nodes)
	tree.output()

	depth_finish = tree.depth() - 1
	leaves_finish = usedcount
	if verbose:
		print "----------"
		print "Depth:", depth_finish
		print "Members:", tree.members()
		print "Leaves:", len(leaves)
		print "Allocated:", leaves_finish

	if depth_start != depth_finish:
		print "[ ERROR: Depth Mismatch! ]"
		sys.exit()

	if leaves_start != leaves_finish or len(leaves) != leaves_finish:
		print "[ ERROR: Leaves Mismatch! ]"
		sys.exit()

print "Saving:", path + "/" + default_output, "   [ 100.00 % ]"
	
outputfile.close()
