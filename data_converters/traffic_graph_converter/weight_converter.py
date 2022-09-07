"""This module handles weight conversion """

import abc
import csv 
import h5py
import igraph as ig
import numpy as np
import os
import pickle


class WeightConverter():
    """
    Base class for the weight converters. They convert DCRNN weights into a
    format compatible for their respective models
    """
    def __init__(self, input_h5_path, input_weight_path, output_weight_path):
        """
        Establish some class variables and read the dataset
        
        Args:
            input_h5_path(Path): path to the input h5 file from DCRNN
            input_weight_path (Path): path to the input .csv file from DCRNN
            output_weight_path (Path): path to the output directory
        """
        self.input_weight_path = input_weight_path
        self.output_weight_path = output_weight_path
        
        # Create the directory if not exists
        if not os.path.exists(self.output_weight_path): 
            os.makedirs(self.output_weight_path)
        
        # For the h5 file, we only need to find the ordered list of bus stop IDs
        with h5py.File(input_h5_path, "r") as f:
            # Different keys for PEMS-BAY and METR-LA
            try:
                self.node_list = list(f['df']['axis0'])
            except KeyError:
                self.node_list = list(f['speed']['axis0'])
        
        # Different node ID datatypes for PEMS-BAY and METR-LA
        try:
            self.node_list = [x.decode("utf-8") for x in self.node_list]
        except AttributeError:
            self.node_list = [str(x) for x in self.node_list]
        
        
    def write_files(self, weights, out_path):
        """
        Write the weights to the output file
        
        Args:
            weights  (numpy array): numpy array containing the weights
            out_path (pathlib Path): path to the output directory
        """
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        np.save(out_path / "weights", train)
        
        
    @abc.abstractmethod 
    def convert_and_write_weights(self):
        """
        Convert the DCRNN weights to a format compatible with various baseline
        models used.
        """
        return 
        
        
    @abc.abstractmethod
    def read_weights(self):
        """
        Read the input weights file to 
        """


class WCVeritasYin(WeightConverter):
    """
    Convert the DCRNN weights to the "veritasyin" format
    
    Paper: https://arxiv.org/pdf/1709.04875v4.pdf
    Code: https://github.com/VeritasYin/STGCN_IJCAI-18
    """
    
    def read_weights(self):
        """
        Read the input weights from the .csv data
        
        Returns:
            csv_data (list): List of the input weights data
        Return:
        """
        # Read the .csv file and store the contents in a list. Remove the header
        csv_data = []
        with open(self.input_weight_path) as f:
            reader = csv.reader(f, delimiter=",")
            for row in reader:
                try:
                    row[-1] = float(row[-1]) 
                    csv_data.append(row)
                except ValueError:
                    continue
        return csv_data
    
    
    def convert_and_write_weights(self):
        """
        Converts the DCRNN weights to the veritasyin format
        """
        # The DCRNN weights store only adjacent stops' weight
        # However, veritasyin's weights store the weights for all node pairs
        # To convert DCRNN to veritasyin, we need to build a graph
        csv_data = self.read_weights()
        G = ig.Graph.TupleList(csv_data, weights=True, directed=False)
        
        # Get shortest distance and create a dictionary for easy querying based
        # on node ID
        dists = G.shortest_paths(weights="weight")
        graph_nodes = G.vs()["name"] 
        
        # Create a dictionary for the distances
        dists_dict = {}
        for iu, u in enumerate(graph_nodes):
            for iv, v in enumerate(graph_nodes):
                uvdist = dists[iu][iv]
                try:
                    uvdist = round(uvdist)
                except OverflowError:
                    pass  
                try:
                    dists_dict[u][v] = uvdist
                except KeyError:
                    dists_dict[u] = {v: uvdist}
        
        # Convert the dictionary to a list of lists to make it easy to print
        out_list = []
        for u in self.node_list:
            u_list = []
            for v in self.node_list:
                u_list.append(str(dists_dict[u][v]))
            out_list.append(u_list)

        # Print the file
        out_file = self.output_weight_path / "weights.csv"
        with open(self.out_file, "w") as f:
            for l in out_list:
                lstr = str(l)
                lstr = lstr.replace("[","")
                lstr = lstr.replace("]","")
                lstr = lstr.replace("'","")
                lstr += "\n"
                f.write(lstr)
                
                
                
class WCJianzhongQi(WeightConverter):
    """
    Convert the DCRNN weights to the "jianzhongqi" format
    
    Paper: https://people.eng.unimelb.edu.au/jianzhongq/papers/TKDE2022_GAMCN.pdf
    Code: https://github.com/alvinzhaowei/GAMCN
    """
    def read_weights(self):
    
        """
        Read the input weights from the .csv data
        
        Returns:
            csv_data (list): List of the input weights data
        Return:
        """
        return pickle.load(open(self.input_weight_path,"rb"),encoding="latin_1")
    
    
    def convert_and_write_weights(self):
        """
        Converts the DCRNN weights to the jianzhongqi format
        """
        weights = self.read_weights()
        
        # The weights used in the DCRNN stores in a way that keeps and uses the
        # traffic detector IDs. On the other hand, the jianzhongqi format 
        # does not store these IDs and just use an integer index starting from 0
        # If the ordering of the nodes in the weights file is the same as in 
        # the .h5 file, then we can just use the weights directly, but we need
        # to check that first
        weights_nodes = weights[0]
        for i in range(len(weights_nodes)):
            if not weights_nodes[i] == self.node_list[i]:
                assert False, "Node ordering is different"
         
        string_list = []
        for i in range(len(weights[2])):
            for j in range(len(weights[2][i])):
                string_list.append("%d %d %f\n" % (i,j,weights[2][i][j]))
        
        out_file = self.output_weight_path / "weights.txt"
        with open(out_file, "w") as f:
            f.writelines(string_list)
                
        