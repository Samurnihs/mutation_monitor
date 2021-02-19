import Bio
from Bio import SeqIO
from Bio.Seq import Seq 
from Bio import AlignIO
import multiprocessing as mp
import numpy as np
import argparse


def tokenseq(seq): # returns sequence of numerical tokens for single sequences
    seq = seq.seq
    tokenseq = np.zeros(len(seq)) # empty array for tokens
    dictionary = {'-': 0, # symbol/token dictionary 
                  'A': 1, 
                  'a': 1, 
                  'T': 2, 
                  't': 2, 
                  'G': 3, 
                  'g': 3, 
                  'C': 4, 
                  'c': 4, 
                  'N': 5, 
                  'n': 5}
    for i in range(len(seq)): # giong through sequence and filling array with tokens
        if seq[i] in dictionary:
            tokenseq[i] = dictionary[seq[i]]
        else:
            tokenseq[i] = 6 # for symbols are not in dictionary
    return tokenseq # returns sequence of tokens


def tokenize(aln): # tokenizes all alignment
    pool = mp.Pool(mp.cpu_count()) # the process uses all cores
    array = np.array(pool.map(tokenseq, aln))
    return array # numpy array of tokens


# The rules of processing tokens. Returns 1 if there is difference between nucleotides. 
# It doesn't take into account other values like 'N', '-' etc.
def diff(a, b): 
    if (b == 0): # absense of base is not SNP
        return 0
    elif (b < 5) and (a-b != 0): # unknown or non-standard base is not correct SNP
        return 1 # Returns 1 in case of SNP. Allows to count SNPs.
    else:
        return 0


def diffseq(seq1, seq2): # applies the diff function to pair of sequences
    return np.array(list(map(lambda x, y: diff(x, y), seq1, seq2)))


def diffref(aln): # Processes diffseq functions for first and all other sequences. The first sequence is reference. 
    return np.array(list(map(lambda x: diffseq(aln[0], x), aln[1: len(aln)])))


# parsing arguments
parser = argparse.ArgumentParser(description='For every position of SARS-CoV-2 genome returns number of discrepancies.')
parser.add_argument('input', type=str, help='Path to alignment fasta file.')
parser.add_argument('output', type=str, help='Path to output array  txt file.')
args = parser.parse_args()

# reading alignment file
input_handle = open(args.input, "rU")
alignment = AlignIO.read(input_handle, "fasta")
input_handle.close()


# column summation and saving result to txt file
np.savetxt(args.output, np.sum(diffref(tokenize(alignment)), axis=0)) 
