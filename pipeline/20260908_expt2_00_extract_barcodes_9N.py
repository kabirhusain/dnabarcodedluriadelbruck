### Extracting Barcodes and Pheromones from Fastq Files

import regex
import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from collections import defaultdict
from Bio import SeqIO
from Bio.Seq import Seq

#### F1 + R1

# Set up regex patterns
umilength_1 = 9
umilength_2 = 8

# Read 1 Regex
UMI = "([A|T|G|C]{" + str(umilength_1) + "})"
primer = "(GGATCCGATCATGCTT){e<=2}"
barcode = "([A|T|G|C]{4}(TT)[A|T|G|C]{4}(TT)[A|T|G|C]{4})"
threeprime = "(GGTACCGCTGATTAGTTATC){e<=2}"
f1r1_read1 = regex.compile(UMI + primer + barcode + threeprime)

# Read 2 Regex
UMI = "([A|T|G|C]{" + str(umilength_2) + "})"
primer = "(CAGAAAACGTCATGGAG){e<=2}"
f1r1_read2 = regex.compile(UMI + primer)

#### F2 + R2

# Set up regex patterns
umilength_1 = 9
umilength_2 = 8

# Read 1 Regex
UMI = "([A|T|G|C]{" + str(umilength_1) + "})"
primer = "(GGATAAAATGTGATAACTAATCAGCGGTACC){e<=4}"
barcode = "([A|T|G|C]{4}(AA)[A|T|G|C]{4}(AA)[A|T|G|C]{4})"
threeprime = "(AAGCATGATCGGATCCGTG){e<=2}"
f2r2_read1 = regex.compile(UMI + primer + barcode + threeprime)

# Read 2 Regex
UMI = "([A|T|G|C]{" + str(umilength_2) + "})"
primer = "(AAACAATCAATGCCAGAG){e<=2}"
f2r2_read2 = regex.compile(UMI + primer)

#### Functions

# Helper function for nested defaultdicts that can be pickled
def default_dict_int():
    return defaultdict(int)

# Fast function for reverse complement
def rc(sequence):
    """Fast reverse complement."""
    complement_dict = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(complement_dict[base] for base in reversed(sequence))

# Function to process raw fastq files and extract barcode and pheromone data
def process_fastq_files(datafiles, r1regex, r2regex, revcomp = False, r1 = "_R1", r2 = "_R2"):
    """Process fastq files to extract barcodes and variants."""

    data_dir = datafiles["data_dir"]
    data_prefix = datafiles["data_prefix"]
    fastq_list = datafiles["fastq_list"]
    data_postfix = datafiles["data_postfix"]

    results = {}

    # Dictionary of dictionaries: bcdCounts[barcode][variant_id] = count
    bcdCounts = defaultdict(default_dict_int)
    umi_ID = defaultdict(int)

    allseq = 0
    bcdseq = 0
    nvariants = 0
    
    for fastq in fastq_list:
        print(f"Starting {data_prefix + fastq}")

        fname = data_prefix + fastq + r1 + data_postfix
        fname2 = data_prefix + fastq + r2 + data_postfix
        
        # Process the fastq files
        with gzip.open(os.path.join(data_dir, fname), "rt") as handle, \
             gzip.open(os.path.join(data_dir, fname2), "rt") as handle2:
            for record, record2 in zip(SeqIO.parse(handle, "fastq"), SeqIO.parse(handle2, "fastq")):
                allseq = allseq + 1
                m = r1regex.match(str(record.seq))
                m2 = r2regex.match(str(record2.seq))
                if m and m2:
                    bcd = m.groups()[2][0:4] + m.groups()[2][6:10] + m.groups()[2][12:16]

                    if bcdseq < 10:
                        print("\t", bcd)

                    bcdseq = bcdseq + 1
                    
                    if revcomp:
                        bcd = rc(bcd)

                    umi = m.groups()[0] + m2.groups()[0]
                    
                    if umi_ID[umi] == 0:
                        nvariants += 1
                        umi_ID[umi] = nvariants
                    
                    # Use barcode string directly as key
                    bcdCounts[bcd][umi_ID[umi]] += 1
                
                # if allseq > 10000:
                #     break

        print(f"Processed {fastq}")
        print(f"\t Total reads so far: {allseq}")
        print(f"\t Reads passing quality control: {bcdseq}")
        print(f"\t i.e. {100*bcdseq/allseq:.2f}%")
        
    results = {
        'bcdCounts': bcdCounts,
        'umi_ID': umi_ID,
        'allseq': allseq,
        'bcdseq': bcdseq
    }
    
    return results


dlist = ["D771-D572", "D772-D573", "D773-D574", "D774-D575", "D775-D576","D776-D577"]

for d5d7 in dlist:
    datfiles = {
        "data_dir": f"Data/",
        "data_prefix": f"AM-LF-pl-LF-01-{d5d7}",
        "fastq_list": [""],
        "data_postfix": ".fastq.gz"
    }

    data_1 = process_fastq_files(datfiles, f1r1_read1, f1r1_read2)
    # data_2 = process_fastq_files(datfiles, f2r2_read1, f2r2_read2, revcomp=True)

    # Make folder for output if it doesn't exist
    output_dir = "Extracted Barcodes"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write the processed data to a pickle file
    output_file = f"{output_dir}/{d5d7}_f1r1.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(data_1, f)

    # # Write the processed data to a pickle file
    # output_file = f"{output_dir}/{d5d7}_f2r2.pkl"
    # with open(output_file, 'wb') as f:
    #     pickle.dump(data_2, f)