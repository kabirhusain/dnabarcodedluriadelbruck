import regex
import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from Bio import SeqIO
from Bio.Seq import Seq
import concurrent.futures


# Helper function for nested defaultdicts that can be pickled
def default_dict_int():
    return defaultdict(int)

# Optimized function to calculate hamming distance
def hamming_distance(s1, s2):
    """Calculate Hamming distance between two DNA sequences efficiently."""
    if len(s1) != len(s2):
        raise ValueError("Sequences must be of equal length")
    
    # Use direct character comparison - much faster than bit operations for small strings
    return sum(c1 != c2 for c1, c2 in zip(s1.upper(), s2.upper()))

# Function to merge similar barcodes
def merge_similar_barcodes(data, max_distance=2):
    """Merge barcodes within Hamming distance of max_distance."""
    bcdCounts = data['bcdCounts']
    
    # Save original read counts for plotting
    read_counts_before = [sum(bcdCounts[b].values()) for b in bcdCounts]
    
    # Sort barcodes by total read counts in descending order
    sorted_barcodes = sorted(bcdCounts.keys(), key=lambda b: sum(bcdCounts[b].values()), reverse=True)
    
    # Merge barcodes within Hamming distance of max_distance
    merged_barcodes = {}
    for i, barcode in enumerate(sorted_barcodes):
        if barcode in merged_barcodes:
            continue  # Skip already merged barcodes
        for other_barcode in sorted_barcodes[i + 1:]:
            if other_barcode in merged_barcodes:
                continue
            if hamming_distance(barcode, other_barcode) <= max_distance:
                # Merge other_barcode into barcode
                for umi_id, count in bcdCounts[other_barcode].items():
                    bcdCounts[barcode][umi_id] += count
                del bcdCounts[other_barcode]
                merged_barcodes[other_barcode] = barcode
    
    # Compute number of umis for each merged barcode
    umiCounts = defaultdict(int)
    for barcode in bcdCounts:
        umiCounts[barcode] = len(bcdCounts[barcode])

    # Compute total read counts for each merged barcode
    totalReads = defaultdict(int)
    for barcode in bcdCounts:
        totalReads[barcode] = sum(bcdCounts[barcode].values())
    
    # Calculate read counts after merging
    read_counts_after = [sum(bcdCounts[b].values()) for b in bcdCounts]

    print(f"\t Total barcodes before merging: {len(sorted_barcodes)}")
    print(f"\t Total barcodes after merging: {len(bcdCounts)}")
    
    return umiCounts,totalReads


# Make folder for output if it doesn't exist
output_dir = "Merged counts"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_file(args):
    d5d7, readdir = args
    print(f"Processing {d5d7} {readdir}")
    output_file = f"Extracted Barcodes/{d5d7}_{readdir}.pkl"
    with open(output_file, 'rb') as f:
        data = pickle.load(f)
    umiCounts, totalReads = merge_similar_barcodes(data, max_distance=1)
    return (f"{d5d7}_{readdir}", {'umiCounts': umiCounts, 'totalReads': totalReads})

# Remove D780-D581
dlist = ["D765-D566", "D766-D567", "D767-D568", "D768-D569", "D769-D570","D770-D571", "D771-D572", "D772-D573", "D773-D574", "D774-D575", "D775-D576","D776-D577", "D777-D578", "D778-D579", "D779-D580", "D780-D581", "D781-D582","D782-D583"]
# dlist = ["D765-D566", "D766-D567", "D767-D568", "D768-D569", "D769-D570","D770-D571", "D771-D572", "D772-D573", "D773-D574", "D774-D575", "D775-D576","D776-D577", "D777-D578", "D778-D579", "D779-D580", "D781-D582","D782-D583"]

# Prepare argument list
tasks = [(d5d7, readdir) for d5d7 in dlist for readdir in ["f1r1"]]

allcounts = {}
with concurrent.futures.ProcessPoolExecutor() as executor:
    for key, result in executor.map(process_file, tasks):
        allcounts[key] = result

# Save all counts to a pickle file
output_file = f"{output_dir}/20250725_merged_counts.pkl"
with open(output_file, 'wb') as f:
    pickle.dump(allcounts, f)
