import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import tables
import logging
import itertools
import operator
from bisect import bisect_left, bisect_right

from typing import Set, List, Dict, Union


def encode(seq: str):
    '''Return the string representation of a DNA sequence
    encoded by 10x Genomics with the 2-bit encoding
    {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    where the most significant bit is set if the sequence
    contained an 'N'
    :seq: the DNA sequence as a string
    Returns
    :num: the 2-bit encoded integer representation of the DNA sequence
    https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/output/molecule_info
    '''
    num = 0
    # look-up for mapping
    lookup = {'A':0, 'C':1, 'G':2, 'T':3}
    # step through the number two bits at a time
    for i in range(0,len(seq),1):
        num = (num << 2) # shift num two bits over
        num = (num | lookup[seq[i]]) # set two lsbs
    return num


def decode(num: int, length):
    '''Return the string representation of a DNA sequence
    encoded by 10x Genomics with the 2-bit encoding
    {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    where the most significant bit is set if the sequence
    contained an 'N'
    :num: the uint32 encoding the sequence
    :length: the number of bases encoded by num
    Returns
    :seq: the string representation of the DNA sequence
    https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/output/molecule_info
    '''
    seq = ''
    # look-up for mapping
    lookup = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    # convert numpy-wrapped ints to python native
    try:
        num = num.item()
    except:
        pass
    # figure out how many charaters this encodes
    if length is None:
        numlength = max(2,num.bit_length()) # at least 2 bits
    else:
        numlength = length*2
    # step through the number two bits at a time
    numshift = num
    for i in range(0,numlength-1,2):
        two_lsbs = numshift & 3
        seq = lookup[two_lsbs] + seq # 3' end is lsb, so keep 5' first
        numshift = (numshift >> 2) # shift num two bits over
    return seq


class MoleculeInfo:
    def __init__(self, mol_h5_path, cr_version, genes_tsv_path=None, extended=False):
        assert cr_version in ['v1', 'v2', 'v3']
        if cr_version == 'v1':
            assert genes_tsv_path is not None

        logging.warning(f'Loading molecule info HDF5 file (CellRanger version: {cr_version})...')
        mol_h5_tab = tables.open_file(mol_h5_path)
        
        # data
        if cr_version in ['v1', 'v2']:
            self.gene_array = mol_h5_tab.root.gene.read()
            self.barcode_array = mol_h5_tab.root.barcode.read()
            self.reads_array = mol_h5_tab.root.reads.read()
            if extended:
                logging.warning('Extended mode: loading H5 additional columns in memory...')
                self.umi_array = mol_h5_tab.root.umi.read()
                self.conf_mapped_uniq_read_pos_array = mol_h5_tab.root.conf_mapped_uniq_read_pos.read()
                self.nonconf_mapped_reads_array = mol_h5_tab.root.nonconf_mapped_reads.read()
                self.unmapped_reads_array = mol_h5_tab.root.unmapped_reads.read()

        if cr_version == 'v3':
            self.gene_array = mol_h5_tab.root.feature_idx.read()
            self.barcode_whitelist = np.asarray(list(
                map(lambda barcode_bytes: encode(barcode_bytes.decode('ascii')),
                    mol_h5_tab.root.barcodes.read())))
            self.barcode_array = self.barcode_whitelist[mol_h5_tab.root.barcode_idx.read()]            
            self.reads_array = mol_h5_tab.root.count.read()
            if extended:
                logging.warning('Extended mode: loading H5 additional columns in memory...')
                self.umi_array= mol_h5_tab.root.umi.raed()
                self.conf_mapped_uniq_read_pos_array = None
                self.nonconf_mapped_reads_array = None
                self.unmapped_reads_array = None

        # gene names
        if cr_version == 'v1':
            gene_names_tsv_pd = pd.read_csv(genes_tsv_path, delimiter='\t', header=None)
            self.gene_names_array = gene_names_tsv_pd.values[:, 0].astype(str)
            self.gene_ids_array = self.gene_names_array
            self.unmappable_gene_idx = len(gene_names_array)
        elif cr_version == 'v2':
            self.gene_names_array = mol_h5_tab.root.gene_names.read()
            self.gene_ids_array = mol_h5_tab.root.gene_ids.read()
            self.unmappable_gene_idx = len(self.gene_names_array)
        elif cr_version == 'v3':
            self.gene_names_array = mol_h5_tab.root.features.name.read()
            self.gene_ids_array = mol_h5_tab.root.features.id.read()
            self.unmappable_gene_idx = len(self.gene_names_array) #### check this
            
        if extended:
            logging.warning('Sorting by (BC, UMI)...')
            # sort molecules hierarchically by (barcode, umi); this will help us in fast binary search lookup of molecules
            sort_index = sorted(
                list(range(len(self.barcode_array))),
                key=lambda idx: (self.barcode_array[idx], self.umi_array[idx]))
            self.gene_array = self.gene_array[sort_index]
            self.barcode_array = self.barcode_array[sort_index]
            self.umi_array = self.umi_array[sort_index]
            self.reads_array = self.reads_array[sort_index]
            
            if cr_version in {'v1', 'v2'}:
                self.conf_mapped_uniq_read_pos_array = self.conf_mapped_uniq_read_pos_array[sort_index]
                self.nonconf_mapped_reads_array = self.nonconf_mapped_reads_array[sort_index]
                self.unmapped_reads_array = self.unmapped_reads_array[sort_index]
        else:
            logging.warning('Extended mode disabled: molecules may NOT be hierarchically sorted by (BC, UMI)!')
        
        # total umi by barcode
        logging.warning('Ranking barcodes by UMI and read count...')
        sorted_barcode_reads = zip(self.barcode_array, self.reads_array)
        self.total_umi_per_barcode = dict()
        self.total_reads_per_barcode = dict()
        group_iter = itertools.groupby(sorted_barcode_reads, operator.itemgetter(0))
        for barcode, grouper in group_iter:
            c_reads = [elem[1] for elem in grouper]
            self.total_umi_per_barcode[barcode] = sum(1 for read in c_reads if read > 0)
            self.total_reads_per_barcode[barcode] = sum(c_reads)

        # sort by total umi
        self.total_umi_per_barcode_list = list(
            (barcode, total_umi_count) for barcode, total_umi_count in self.total_umi_per_barcode.items())
        self.sorted_total_umi_per_barcode_list = sorted(
            self.total_umi_per_barcode_list, key=operator.itemgetter(1), reverse=True)

        # sort by total reads
        self.total_reads_per_barcode_list = list(
            (barcode, total_read_count) for barcode, total_read_count in self.total_reads_per_barcode.items())
        self.sorted_total_reads_per_barcode_list = sorted(
            self.total_reads_per_barcode_list, key=operator.itemgetter(1), reverse=True)



def get_full_umi_count_statistics(barcode_set: Union[Set[int], None],
                                  gene_array: np.ndarray,
                                  reads_array: np.ndarray,
                                  barcode_array: np.ndarray) -> Dict[int, List[int]]: 
    if barcode_set is not None:
        # subset barcodes
        subset_indices = list(idx for idx, barcode in enumerate(barcode_array) if barcode in barcode_set)
        subset_gene_array = list(gene_array[idx] for idx in subset_indices)
        subset_reads_array = list(reads_array[idx] for idx in subset_indices)
    else:
        subset_gene_array = gene_array
        subset_reads_array = reads_array
    
    # group molecules by gene index
    sorted_gene_reads = sorted(zip(subset_gene_array, subset_reads_array), key=operator.itemgetter(0))

    # aggregate read counts by gene index
    full_count_stats = dict()
    group_iter = itertools.groupby(sorted_gene_reads, operator.itemgetter(0))
    for gene_idx, grouper in group_iter:
        full_count_stats[gene_idx] = np.asarray(list(elem[1] for elem in grouper))
    return full_count_stats


def get_gene_index_sorted_by_expression(full_count_stats: Dict[int, List[int]]) -> List[int]:
    # sort genes according to their observed unique instances
    # in the full UMI counting statistics dictionary
    observed_cdna_per_gene = dict(
        (gene_index, len(read_counts))
        for gene_index, read_counts in full_count_stats.items())
    gene_idx_by_observed_cdna = list(
        elem[0] for elem in sorted(
            list(observed_cdna_per_gene.items()), key=lambda elem: elem[1], reverse=True))
    return gene_idx_by_observed_cdna


def plot_umi_hist_for_gene(ax,
                           gene_idx: int,
                           gene_names_array: np.ndarray,
                           full_count_stats: Dict,
                           min_family_size_frac: int=1,
                           min_family_size_plot: int=0,
                           decode_gene_ascii: bool=True,
                           max_family_size_plot: int=60,
                           hist_bins: int=20):
    family_size_list = full_count_stats[gene_idx]
    family_size_list_for_plotting = [x for x in family_size_list if x > min_family_size_plot]

    gene_name_ascii = (gene_names_array[gene_idx].decode('ascii')
                       if decode_gene_ascii else gene_names_array[gene_idx])
    _ = ax.hist(family_size_list_for_plotting,
        bins=hist_bins,
        range=(0,max_family_size_plot),
                log=False, normed=True, label=gene_name_ascii)
    ax.legend(fontsize=12)
    ax.set_ylabel(r'$P(FS)$', fontsize=18)
    ax.set_xlabel(r'$FS$', fontsize=18)
    
    lte_min_family_size = sum(
        1 for family_size in family_size_list if family_size <= min_family_size_frac)
    all_observations = len(family_size_list)
    
    print(f'Observations with family size <= {min_family_size_frac} '
          f'for gene {gene_name_ascii}: {100*float(lte_min_family_size/all_observations):.1f}%')


def plot_umi_hist_for_genes(gene_names: List,
                            gene_names_array: np.ndarray,
                            full_count_stats: Dict,
                            min_family_size_frac: int=1,
                            min_family_size_plot: int=0,
                            decode_gene_ascii: bool=True,
                            max_family_size_plot: int=60):
    gene_names_list = gene_names_array.tolist()
    gene_idxs = [gene_names_list.index(gene_name) for gene_name in gene_names]

    axs = plt.subplots(ncols=len(gene_names), figsize=(12,4))
    
    for ax, gene_idx, gene_name in zip(axs[1], gene_idxs, gene_names):
        
        family_size_list = full_count_stats[gene_idx]
        family_size_list_for_plotting = [x for x in family_size_list if x > min_family_size_plot]

        gene_name_ascii = gene_name.decode('ascii') if decode_gene_ascii else gene_name
        
        _ = ax.hist(family_size_list_for_plotting, bins=max_family_size_plot, range=(0,max_family_size_plot),
                    log=False, normed=True, label=gene_name_ascii)
        ax.legend(fontsize=12)
        ax.set_ylabel(r'$P(FS)$', fontsize=18)
        ax.set_xlabel(r'$FS$', fontsize=18)
        
        lte_min_family_size = sum(
            1 for family_size in family_size_list if family_size <= min_family_size_frac)
        all_observations = len(family_size_list)
        
        print(f'Observations with family size <= {min_family_size_frac} '
              f'for gene {gene_name_ascii}: {100*float(lte_min_family_size/all_observations):.1f}%')

    plt.tight_layout()



BASES = ['A', 'C', 'T', 'G']

def generate_HD_1_power_set(input_string, remove_self=True):
    out = set()
    out.add(input_string)
    for pos in range(len(input_string)):
        for base in BASES:
            out.add(input_string[:pos] + base + input_string[(pos + 1):])
    if remove_self:
        out.remove(input_string)
    return out

def generate_HD_k_power_set(input_string, k, remove_self=True):
    out = set()
    out.add(input_string)
    for j in range(k):
        old_out = set(out)
        for input_from_old in old_out:
            out.update(generate_HD_1_power_set(input_from_old, remove_self=True))
    if remove_self:
        out.remove(input_string)
    return out

def generate_indel_1_power_set(input_string, remove_self=True):
    out = set()
    out.add(input_string)
    # remove one base and add one base at the end
    for pos in range(len(input_string)):
        for base in BASES:
            out.add(input_string[:pos] + input_string[(pos + 1):] + base)
    # add one base and remove from the end
    for pos in range(len(input_string)):
        for base in BASES:
            out.add(input_string[:pos] + base + input_string[pos:(len(input_string) - 1)])
    if remove_self:
        out.remove(input_string)
    return out

def generate_power_set(input_string, max_hamming_distance=1, include_indel=True, remove_self=False):
    out = set()
    out.add(input_string)
    if max_hamming_distance > 0:
        out.update(generate_HD_k_power_set(input_string, max_hamming_distance))
    if include_indel:
        out.update(generate_indel_1_power_set(input_string))
    if remove_self:
        out.remove(input_string)
    return out

def generate_encoded_power_set(encoded_barcode, encoded_umi,
                               max_hamming_distance=1, include_indel=True, remove_self=False):
    decoded_barcode = decode(encoded_barcode, length=16)
    decoded_umi = decode(encoded_umi, length=10)
    total_barcode = decoded_barcode + decoded_umi
    decoded_power_set = generate_power_set(total_barcode, max_hamming_distance, include_indel, remove_self)
    return set((encode(mutated[:16]), encode(mutated[16:])) for mutated in decoded_power_set)



def get_molecules(barcode, umi, mol_info: MoleculeInfo):
    """Returns the index of all molecules with given (encoded) barcode and umi. If the molecule
    does not exist, returns an empty list."""
    left_barcode_index = bisect_left(mol_info.barcode_array, barcode)
    right_barcode_index = bisect_right(mol_info.barcode_array, barcode)
    if left_barcode_index == right_barcode_index:  # barcode does not exist
        return []
    umi_array_for_barcode = mol_info.umi_array[left_barcode_index:right_barcode_index]
    left_umi_index = bisect_left(umi_array_for_barcode, umi)
    right_umi_index = bisect_right(umi_array_for_barcode, umi)
    if left_umi_index == right_umi_index:  # umi does not exist
        return []
    return [left_barcode_index + j for j in range(left_umi_index, right_umi_index)]


def get_all_barcode_umi_for_gene_index(gene_index, mol_info: MoleculeInfo):
    indices = mol_info.gene_array == gene_index
    barcode_array_for_gene = mol_info.barcode_array[indices]
    umi_array_for_gene = mol_info.umi_array[indices]
    reads_array_for_gene = mol_info.reads_array[indices]
    return {(barcode, umi): reads
            for barcode, umi, reads in zip(barcode_array_for_gene, umi_array_for_gene, reads_array_for_gene)}


