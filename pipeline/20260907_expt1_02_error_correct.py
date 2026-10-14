"""

Implements the following:

1. Orders all putative barcodes from most frequent to least frequent
2. Starting from the most frequent barcode, find all n-neighbours that:
    a. have fewer counts
    b. have not yet been merged
3. Merge all n-neighbours of that barcode together, and count unique UMIs seen

"""

import numpy as np
import time
import os

from scipy.sparse import load_npz, dok_array

def encodeATGC(bcdStr):
    return int(bcdStr.replace("A","0").replace("T","1").replace("G","2").replace("C","3"), 4)

def decodeATGC(bcdInt, bcdLength = 12):
    bcdString = np.base_repr(bcdInt, base = 4)
    bcdString = ("0"*(bcdLength-len(bcdString)) ) + bcdString
    return bcdString.replace("0","A").replace("1","T").replace("2","G").replace("3","C")

def intToOneHotATGC(bcdInt, bcdLength = 12):
    bcdString = np.base_repr(bcdInt, base = 4)
    bcdString = ("0"*(bcdLength-len(bcdString)) ) + bcdString
    
    nucleotide_encoding = {'0': 0, '1': 1, '2': 2, '3': 3}
    encoding = np.zeros(bcdLength * 4, dtype=int)
    
    # Create an array of indices corresponding to the nucleotides in the sequence
    indices = np.array([nucleotide_encoding[n] for n in bcdString])
    
    # Use advanced indexing to set the appropriate elements to 1
    encoding[np.arange(bcdLength) * 4 + indices] = 1
    
    return encoding



mergeDist = 1 # Hamming distance to merge
bcdLength = 12 # How big is the barcode?
batchSize = 128 # How many Hamming distance calculations should I do at once? 128 seems to work well.

loadPrefix = "../20230207_bcdExtract/"
savePrefix = "./"


datFile = "D709_D501"


#### Load and pre-process data


t0 = time.time()

datUMI = load_npz(loadPrefix + datFile + '_bcdUMI.npz')
dat = load_npz(loadPrefix + datFile + '_bcd.npz')
counts = np.array([dat[i,0] for i in dat.nonzero()[0]])
bcds = np.array([i for i in dat.nonzero()[0]])

t1 = time.time()

print("Loaded " + str(np.sum(counts)) + " reads and extracted barcodes in " + str(t1-t0) + "s")
print("Making a one-hot matrix...")

# Convert to a one-hot matrix
oneHotBcds = np.zeros([len(bcds), 12*4])

t0 = time.time()

for whichbcd in np.arange(len(bcds)):
    oneHotBcds[whichbcd, :] = intToOneHotATGC(bcds[whichbcd])

t1 = time.time()

total = t1-t0
print("Converted " + str(len(bcds)) + " barcodes to a one hot matrix in " + str(total) + "s")


# Sort barcodes

t0 = time.time()

ordered = np.flip(np.argsort(counts)) # from largest to smallest.

t1 = time.time()

print("Barcodes sorted by read counts in " + str(t1-t0) + "s")





##### Merge barcodes

"""

OK, merge barcodes that are within a certain Hamming distance of each other.

The way this is going to work is the following. I am going to take each barcode one by one 
and then project it onto the one-Hot Matrix to get the Hamming distance.

I will find those that are within mergeDist of each other and collapse into the 
barcode with the most counts.

"""

rawBcdNum = len(bcds)
batchNum = int(np.ceil(rawBcdNum/batchSize)) # When processing all the data, this should be int(np.ceil(rawBcdNum/batchSize))

correctedBcdCounts = dok_array((4**bcdLength, 1), dtype=np.uint64) # The final matrix


# Resort
orderedOneHot = oneHotBcds[ordered,:]
orderedBcds = bcds[ordered]

notmerged = [True]*len(bcds) # Keep track of already merged.
oneHotThresh = bcdLength - mergeDist

t0 = time.time()

totalReads = 0
for batch in np.arange(batchNum):
    startingBcd = batch*batchSize
    endingBcd = min((batch+1)*batchSize, rawBcdNum)
    
    distanceMatrix = np.dot(orderedOneHot, orderedOneHot[startingBcd:endingBcd,:].transpose())
    
    for whichbcd in np.arange(startingBcd, endingBcd):
        if notmerged[whichbcd]:
            withinHamming = np.where(distanceMatrix[:,whichbcd - startingBcd] >= oneHotThresh)[0]
            
            UMIs = np.array([])
            for whichMergedBcd in withinHamming:
                UMIs = np.append(UMIs, datUMI[ orderedBcds[whichMergedBcd] ].nonzero()[1])
                notmerged[whichMergedBcd] = False # signify merged

            correctedBcdCounts[ orderedBcds[whichbcd] , 0 ] = len(np.unique(UMIs))
            totalReads = totalReads + correctedBcdCounts[ orderedBcds[whichbcd] , 0 ]
                    
t1 = time.time()
print("Error corrected " + str(batchNum*batchSize) + " barcodes in " + str(t1 - t0) + "s")


print("Finished merging. Final number of unique barcodes: " + str(correctedBcdCounts.nnz))
print("Total UMI-corrected reads: " + str(totalReads))
save_npz(savePrefix + datFile + "_errorCorrectedBcds.npz", correctedBcdCounts.tocsr())







# import regex
# import gzip
# from Bio import SeqIO

# from scipy.sparse import dok_array, save_npz
# import numpy as np

# def encodeATGC(bcdStr):
#     return int(bcdStr.replace("A","0").replace("T","1").replace("G","2").replace("C","3"), 4)


# def decodeATGC(bcdInt, bcdLength = 12):
#     bcdString = np.base_repr(bcdInt, base = 4)
#     bcdString = ("A"*(bcdLength-len(bcdString)) ) + bcdString
#     return bcdString.replace("0","A").replace("1","T").replace("2","G").replace("3","C")



# bcdlength = 12
# umilength_1 = 8
# umilength_2 = 6


# UMI = "([A|T|G|C]{" + str(umilength_1) + "})"
# primer = "(GGATCCGATCATGCTT){e<=3}"
# barcode = "([A|T|G|C]{4}(TT){e<=1}[A|T|G|C]{4}(TT){e<=1}[A|T|G|C]{4})"
# threeprime = "(GGTACCGCTGA){e<=2}"

# primer2 = "(CAGAAAA){e<=2}"

# read1regex = regex.compile(UMI + primer + barcode + threeprime)
# read2regex = regex.compile("([A|T|G|C]{" + str(umilength_2) + "})" + primer2)



# dataDir = "/scratch/midway2/kabirh/202302Can1HiDenSeq/"
# fastqList = ["D709_D501_S65", "D709_D503_S67", "D709_D505_S69"]

# for fastq in fastqList:
#     print("Starting " + fastq)
#     fname = fastq + '_R1_001.fastq.gz'
#     fname2 = fastq + '_R2_001.fastq.gz'

#     bcdUMICounts = dok_array((4**bcdlength, 4**(umilength_1 + umilength_2)), dtype=np.uint64)
#     bcdCounts = dok_array((4**bcdlength, 1), dtype=np.uint64)

#     allseq = 0
#     bcdseq = 0

#     with gzip.open(dataDir + fname, "rt") as handle, gzip.open(dataDir + fname2, "rt") as handle2:
#         for record,record2 in zip(SeqIO.parse(handle, "fastq"),SeqIO.parse(handle2, "fastq")):
#             allseq = allseq + 1
#             m = read1regex.match(str(record.seq))
#             m2 = read2regex.match(str(record2.seq))
#             if m and m2:
#                 bcd = m.groups()[2][0:4] + m.groups()[2][6:10] + m.groups()[2][12:16] 
#                 bcdseq = bcdseq + 1
                
#                 umi = m.groups()[0] + m2.groups()[0]
                
#                 bcdUMICounts[encodeATGC(bcd), encodeATGC(umi)] = bcdUMICounts[encodeATGC(bcd), encodeATGC(umi)] + 1
#                 bcdCounts[encodeATGC(bcd), 0] = bcdCounts[encodeATGC(bcd), 0] + 1
                
#     print(fastq)            
#     print("Total reads: " + str(allseq))
#     print("Reads passing quality control: " + str(bcdseq))
#     print("i.e. " + str(100*bcdseq/allseq) + "%")

#     save_npz(fastq.split("S")[0] + "bcdUMI.npz",bcdUMICounts.tocsr())
#     save_npz(fastq.split("S")[0] + "bcd.npz",bcdCounts.tocsr())
