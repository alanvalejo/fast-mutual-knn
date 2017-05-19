#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MkNN (Mutual k Nearest Neighbor Graph Construction)
=====================================================

Copyright (C) 2016 Alan Valejo <alanvalejo@gmail.com> All rights reserved
Copyright (C) 2016 Thiago Faleiros <thiagodepaulo@gmail.com> All rights reserved

For the graph sparsification, k-Nearest Neighbors (MkNN) is a method usually
applied. Each vertex is associated with a set of k closest vertices (Nk)
according to a similarity criterion. A variation of this method is mutual
kNN graphs (MkNN), in which there is a connection between two vertices only
if the rule of the neighborhood has been fulfilled by both vertices.

This file is part of Mutual MkNN.

MkNN is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

MkNN is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Mutual MkNN. If not, see <http://www.gnu.org/licenses/>.
"""

import csv
import os
import sys
import argparse
import numpy as np

from multiprocessing import Pipe, Process
from helper import write_ncol, write_pajek
from scipy import spatial

__maintainer__ = 'Alan Valejo'
__author__ = 'Alan Valejo, Thiago Faleiros'
__email__ = 'alanvalejo@gmail.com', 'thiagodepaulo@gmail.com'
__credits__ = ['Alan Valejo', 'Thiago Faleiros']
__homepage__ = 'https://github.com/alanvalejo/mknn'
__license__ = 'GNU'
__docformat__ = 'markdown en'
__version__ = '0.1'
__date__ = '2016-12-01'

def knn(obj_subset, data, kdtree, k, sender):
	"""  K nearest neighbor graph construction.

	Args:
		obj_subset (array): Set of vertices by threads
		data (np.array): Original data table
		kdtree (spatial.KDTree): KD tree accounting for from data
		k (int): K nearest neighbors
		sender (multiprocessing.Connection): Pipe connection objects
	"""

	dic_knn = dict()
	for obj in obj_subset:
		obj_attrs = data[obj]
		# (dists, indexs) = kdtree.query(obj_attrs, k=(k+1))
		dic_knn[obj] = kdtree.query(obj_attrs, k=(k + 1))
		# Considering the first nearst neighbor equal itself
		dic_knn[obj] = (dic_knn[obj][0][1:], dic_knn[obj][1][1:])

	sender.send(dic_knn)

def mutual_knn(obj_subset, k, dic_knn, sender):
	""" Mutual k nearest neighbor graph construction.

	Args:
		obj_subset (array): Set of vertices by threads
		k2 (int): Semi-supervised K
		buff (dictinary): Each vertex is associated with the nearest neighbor labeled
		dic_knn (dictionary): List of Knn to each vertice
		sender (multiprocessing.Connection): Pipe connection objects
	"""

	ew = [] # Set of weighted edges
	for obj in obj_subset:
		obj_knn = dic_knn[obj]
		# For each KNN vertex
		for i, nn in enumerate(obj_knn[1]):
			if obj == nn: continue
			nn_knn = dic_knn[nn]
			# If it is mutual
			if obj in nn_knn[1]:
				# Distance between obj and nn
				d1 = obj_knn[0][i]
				# Tuple (edge, weight)
				ew.append((obj, nn, 1 / (1 + d1)))

	sender.send(ew)

def main():
	""" Main entry point for the application when run from the command line. """

	# Parse options command line
	usage = 'use "%(prog)s --help" for more information'
	description = 'MkNN graph construction'
	parser = argparse.ArgumentParser(description=description, usage=usage, formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=30, width=100))
	optional = parser._action_groups.pop()
	required = parser.add_argument_group('required arguments')
	required.add_argument('-f', '--filename', dest='filename', action='store', type=str, metavar='FILE', default=None, help='name of the %(metavar)s to be loaded')
	optional.add_argument('-d', '--directory', dest='directory', action='store', type=str, metavar='DIR', default=None, help='directory of FILE if it is not current directory')
	optional.add_argument('-o', '--output', dest='output', action='store', type=str, metavar='FILE', default=None, help='name of the %(metavar)s to be save')
	required.add_argument('-l', '--label', dest='label', action='store', type=str, metavar='FILE', default=None, help='list of labels points used to construct RGCLI')
	optional.add_argument('-k', '--k', dest='k', action='store', type=int, metavar='int', default=3, help='kNN (default: %(default)s)')
	optional.add_argument('-t', '--threads', dest='threads', action='store', type=int, metavar='int', default=4, help='number of threads (default: %(default)s)')
	optional.add_argument('-e', '--format', dest='format', action='store', choices=['ncol', 'pajek'], type=str, metavar='str', default='ncol', help='format output file. Allowed values are ' + ', '.join(['ncol', 'pajek']) + ' (default: %(default)s)')
	optional.add_argument('-c', '--skip_last_column', dest='skip_last_column', action='store_false', default=True, help='skip last column (default: true)')
	parser._action_groups.append(optional)
	options = parser.parse_args()

	# Process options and args
	if options.filename is None:
		parser.error('required -f [filename] arg.')
	if options.format not in ['ncol', 'pajek']:
		parser.error('supported formats: ncol and pajek.')
	if options.directory is None:
		options.directory = os.path.dirname(os.path.abspath(options.filename))
	else:
		if not os.path.exists(options.directory): os.makedirs(options.directory)
	if not options.directory.endswith('/'): options.directory += '/'
	if options.output is None:
		filename, extension = os.path.splitext(os.path.basename(options.filename))
		options.output = options.directory + filename + '-mknn' + str(options.k) + '.' + options.format
	else:
		options.output = options.directory + options.output

	# Detect wich delimiter and which columns to use is used in the data
	with open(options.filename, 'r') as f:
		first_line = f.readline()
	sniffer = csv.Sniffer()
	dialect = sniffer.sniff(first_line)
	ncols = len(first_line.split(dialect.delimiter))
	if not options.skip_last_column: ncols -= 1

	# Reading data table
	# Acess value by data[object_id][attribute_id]
	# Acess all attributs of an object by data[object_id]
	# To transpose set arg unpack=True
	data = np.loadtxt(options.filename, delimiter=dialect.delimiter, usecols=range(0, ncols))
	attr_count = data.shape[1] # Number of attributes
	obj_count = data.shape[0] # Number of objects
	obj_set = range(0, obj_count) # Set of objects

	# Create KD tree from data
	kdtree = spatial.KDTree(data)

	# Size of the set of vertices by threads, such that V = {V_1, ..., V_{threads} and part = |V_i|
	part = obj_count / options.threads

	# Creating list of labeled nearst neighours
	receivers = []
	for i in xrange(0, obj_count, part):
		# Returns a pair (conn1, conn2) of Connection objects representing the ends of a pipe
		sender, receiver = Pipe()
		p = Process(target=knn, args=(obj_set[i:i+part], data, kdtree, options.k, sender))
		p.daemon = True
		p.start()
		receivers.append(receiver)

	dic_knn = dict()
	for receiver in receivers:
		# Waiting threads
		dic_knn_aux = receiver.recv()
		dic_knn.update(dic_knn_aux)

	# Starting mutual knn processing
	receivers = []
	for i in xrange(0, obj_count, part):
		sender, receiver = Pipe()
		p = Process(target=mutual_knn, args=(obj_set[i:i + part], options.k, dic_knn, sender))
		p.daemon = True
		p.start()
		receivers.append(receiver)

	# Create set of weighted edges
	edgelist = ''
	for receiver in receivers:
		# Waiting threads
		ew = receiver.recv()
		for edge in ew:
			edgelist += '%s %s %s\n' % edge

	# Save edgelist in output file
	if options.format == 'ncol':
		write_ncol(options.output, edgelist)
	else:
		write_pajek(options.output, obj_count, edgelist)

if __name__ == "__main__":
	sys.exit(main())
