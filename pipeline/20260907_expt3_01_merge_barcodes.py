import pickle
import os
import time
from itertools import combinations
from collections import defaultdict
import concurrent.futures


# Helper function for nested defaultdicts that can be pickled
def default_dict_int():
    return defaultdict(int)


DNA_BASES = 'ATGC'


def hamming_neighbors_d1(barcode):
    """Generate all Hamming-distance-1 neighbors of a DNA barcode."""
    neighbors = []
    barcode_list = list(barcode)
    for i in range(len(barcode)):
        orig = barcode_list[i]
        for base in DNA_BASES:
            if base != orig:
                barcode_list[i] = base
                neighbors.append(''.join(barcode_list))
        barcode_list[i] = orig
    return neighbors


def hamming_neighbors_d2(barcode):
    """Generate all Hamming-distance-1 and distance-2 neighbors of a DNA barcode."""
    neighbors = []
    barcode_list = list(barcode)
    blen = len(barcode)

    # Distance 1: 12 * 3 = 36 neighbors
    for i in range(blen):
        orig = barcode_list[i]
        for base in DNA_BASES:
            if base != orig:
                barcode_list[i] = base
                neighbors.append(''.join(barcode_list))
        barcode_list[i] = orig

    # Distance 2: C(12,2) * 9 = 594 neighbors
    for i, j in combinations(range(blen), 2):
        orig_i = barcode_list[i]
        orig_j = barcode_list[j]
        for base_i in DNA_BASES:
            if base_i == orig_i:
                continue
            barcode_list[i] = base_i
            for base_j in DNA_BASES:
                if base_j == orig_j:
                    continue
                barcode_list[j] = base_j
                neighbors.append(''.join(barcode_list))
            barcode_list[j] = orig_j
        barcode_list[i] = orig_i

    return neighbors


def merge_similar_barcodes(data, max_distance=1):
    """Merge barcodes within Hamming distance using O(n) neighbor lookup.

    Instead of comparing all pairs O(n²), we:
    1. Put all barcodes in a set for O(1) lookup
    2. For each barcode (high-count first), generate its Hamming neighbors
    3. Merge any neighbor found in the set into the current barcode

    For 12bp DNA barcodes:
      - distance 1: 36 neighbors per barcode
      - distance 2: 630 neighbors per barcode
    """
    bcdCounts = data['bcdCounts']

    # Pre-compute total reads once per barcode
    total_reads = {b: sum(counts.values()) for b, counts in bcdCounts.items()}

    print(f"\t Total barcodes before merging: {len(bcdCounts)}")

    # Sort barcodes by total read count descending (high-count absorbs low-count)
    sorted_barcodes = sorted(total_reads, key=total_reads.get, reverse=True)

    # Set of barcodes still available to be merged
    active = set(bcdCounts.keys())

    # Choose neighbor generator based on max_distance
    if max_distance == 1:
        get_neighbors = hamming_neighbors_d1
    elif max_distance == 2:
        get_neighbors = hamming_neighbors_d2
    else:
        raise ValueError(f"max_distance must be 1 or 2, got {max_distance}")

    merge_count = 0
    for barcode in sorted_barcodes:
        if barcode not in active:
            continue  # Already merged into a higher-count barcode

        # Find neighbors that exist in the active set
        for neighbor in get_neighbors(barcode):
            if neighbor in active and neighbor != barcode:
                # Merge neighbor into barcode
                for umi_id, count in bcdCounts[neighbor].items():
                    bcdCounts[barcode][umi_id] += count
                del bcdCounts[neighbor]
                active.discard(neighbor)
                merge_count += 1

    # Compute UMI counts and total reads for merged barcodes
    umiCounts = defaultdict(int)
    totalReads = defaultdict(int)
    for barcode in bcdCounts:
        umiCounts[barcode] = len(bcdCounts[barcode])
        totalReads[barcode] = sum(bcdCounts[barcode].values())

    print(f"\t Total barcodes after merging: {len(bcdCounts)} ({merge_count} merged)")

    return umiCounts, totalReads


# Make folder for output if it doesn't exist
output_dir = "Merged counts fast"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def process_file(args):
    d5d7, readdir = args
    key = f"{d5d7}_{readdir}"

    # Checkpointing: skip if already processed
    checkpoint_file = f"{output_dir}/{key}_merged.pkl"
    # if os.path.exists(checkpoint_file):
    #     print(f"Skipping {key} (checkpoint exists)")
    #     with open(checkpoint_file, 'rb') as f:
    #         result = pickle.load(f)
    #     return (key, result)

    print(f"Processing {key}")
    t0 = time.time()

    input_file = f"Extracted Barcodes/{key}.pkl"
    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    umiCounts, totalReads = merge_similar_barcodes(data, max_distance=1)
    result = {'umiCounts': umiCounts, 'totalReads': totalReads}

    # Save checkpoint
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(result, f)

    elapsed = time.time() - t0
    print(f"Finished {key} in {elapsed:.1f}s")

    return (key, result)


dlist = [
    "D756-D560", "D757-D561", "D758-D562", "D759-D563",
    "D761-D565", "D762-D566", "D763-D567", "D764-D568",
    "D707-D506", "D767-D570", "D768-D571", "D769-D572",
]

# Prepare argument list
tasks = [(d5d7, readdir) for d5d7 in dlist for readdir in ["f1r1"]]

allcounts = {}
# Limit to 4 workers to control memory (~4 files loaded at once)
with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(process_file, task): task for task in tasks}
    for future in concurrent.futures.as_completed(futures):
        key, result = future.result()
        allcounts[key] = result

# Save combined output (same format as original for downstream compatibility)
output_file = f"{output_dir}/20260224_merged_counts.pkl"
with open(output_file, 'wb') as f:
    pickle.dump(allcounts, f)

print(f"\nSaved combined results to {output_file}")
print(f"Total files processed: {len(allcounts)}")
