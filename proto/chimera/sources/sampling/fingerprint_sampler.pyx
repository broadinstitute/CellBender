# distutils: language = c++

cimport cython

from cyrandom.cyrandom cimport randint

from cpython cimport bool
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cython.operator cimport dereference as deref, preincrement as inc

from libc.stdint cimport int32_t
from libcpp.unordered_set cimport unordered_set as unordered_set
from libcpp.vector cimport vector as vector

ctypedef int32_t INT_DTYPE
ctypedef float FLOAT_DTYPE

# TODO refactor out
cdef class CSRFloatMatrix:
    cdef int32_t n_rows
    cdef int32_t n_cols
    cdef int32_t* indptr
    cdef int32_t* indices
    cdef FLOAT_DTYPE* data

    def __cinit__(
            self,
            size_t n_rows,
            size_t n_cols,
            int32_t[:] indptr,
            int32_t[:] indices,
            FLOAT_DTYPE[:] data):
        self.n_rows = n_rows
        self.n_cols = n_cols

        # allocate memory
        self.indptr = <int32_t*> PyMem_Malloc((n_rows + 1) * sizeof(int32_t))
        self.indices = <int32_t*> PyMem_Malloc(len(indices) * sizeof(int32_t))
        self.data = <FLOAT_DTYPE*> PyMem_Malloc(len(indices) * sizeof(FLOAT_DTYPE))

        cdef size_t i, j
        assert len(indptr) == n_rows + 1, \
            f"The length of indptr ({len(indptr)}) does not match match n_rows + 1 ({n_rows + 1})"
        assert len(indices) == len(data), \
            f"The length of indices ({len(indices)}) does not match the length of data ({len(data)})"
        assert indptr[0] == 0, \
            "The first entry in indptr must be 0"
        assert indptr[n_rows] == len(indices), \
            "The last entry in indptr must be equal to the length of indices"
        for i in range(n_rows):
            assert indptr[i + 1] >= indptr[i], \
                "indptr must be ascending"
            for j in range(indptr[i], indptr[i + 1] - 1):
                assert indices[j + 1] > indices[j], \
                    "for each row, indices must be unique and sorted in ascending order"
            assert indices[indptr[i + 1] - 1] < n_cols, \
                f"indices must be in range [0, {n_cols})"

        for i in range(n_rows + 1):
            self.indptr[i] = indptr[i]
        for i in range(len(indices)):
            self.indices[i] = indices[i]
        for i in range(len(indices)):
            self.data[i] = data[i]

    cdef int32_t get_non_zero_cols(self, int32_t i_row) nogil:
        return self.indptr[i_row + 1] - self.indptr[i_row]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef void copy_rows_to_dense(self, int32_t[:] selected_row_indices, FLOAT_DTYPE[:, ::1] out):
        cdef Py_ssize_t i, j, n
        cdef Py_ssize_t n_selected_row_indices = len(selected_row_indices)
        for i in range(n_selected_row_indices):
            n = selected_row_indices[i]
            for j in range(self.indptr[n], self.indptr[n + 1]):
                out[i, self.indices[j]] = self.data[j]

    def __dealloc__(self):
        PyMem_Free(self.indptr)
        PyMem_Free(self.indices)
        PyMem_Free(self.data)


# TODO refactor out
cdef class CSRBinaryMatrix:
    cdef int32_t n_rows
    cdef int32_t n_cols
    cdef int32_t* indptr
    cdef int32_t* indices
    
    def __cinit__(
            self,
            size_t n_rows,
            size_t n_cols,
            size_t indices_sz,
            *args, **kwargs):
        self.n_rows = n_rows
        self.n_cols = n_cols

        # allocate memory
        self.indptr = <int32_t*> PyMem_Malloc((n_rows + 1) * sizeof(int32_t))
        self.indices = <int32_t*> PyMem_Malloc(indices_sz * sizeof(int32_t))
    
    def __init__(self,
            size_t n_rows,
            size_t n_cols,
            size_t indices_sz,
            int32_t[:] indptr,
            int32_t[:] indices,
            bool skip_copy = False,
            bool skip_validation = False):
        
        cdef size_t i, j
        if not skip_validation:
            assert len(indptr) == n_rows + 1, \
                f"The length of indptr ({len(indptr)}) does not match match n_rows + 1 ({n_rows + 1})"
            assert len(indices) == indices_sz, \
                f"The length of indices ({len(indices)}) does not match indices_sz ({indices_sz})"
            assert indptr[0] == 0, \
                "The first entry in indptr must be 0"
            assert indptr[n_rows] == indices_sz, \
                "The last entry in indptr must be equal to the length of indices"        
            for i in range(n_rows):
                assert indptr[i + 1] >= indptr[i], \
                    "indptr must be ascending"
                for j in range(indptr[i], indptr[i + 1] - 1):
                    assert indices[j + 1] > indices[j], \
                        "for each row, indices must be unique and sorted in ascending order"
                assert indices[indptr[i + 1] - 1] < n_cols, \
                    f"indices must be in range [0, {n_cols})"

        if not skip_copy:
            for i in range(n_rows + 1):
                self.indptr[i] = indptr[i]
            for i in range(indices_sz):
                self.indices[i] = indices[i]
                
    cdef int32_t get_non_zero_cols(self, int32_t i_row) nogil:
        return self.indptr[i_row + 1] - self.indptr[i_row]

    def __invert__(self) -> CSRBinaryMatrix:
        """Returns the bitwise not of the matrix."""
        # allocate memory
        cdef int32_t inv_indices_sz = self.n_rows * self.n_cols - self.indptr[self.n_rows]
        inv_array = CSRBinaryMatrix(
            n_rows=self.n_rows,
            n_cols=self.n_cols,
            indices_sz=inv_indices_sz,
            indptr=None,
            indices=None,
            skip_copy=True,
            skip_validation=True)

        cdef:
            int32_t first_index, last_index
            int32_t n_values, n_compl_values
            int32_t i_row, i_col, c_index, inv_indices_loc
        
        inv_indices_loc = 0
        inv_array.indptr[0] = 0
        for i_row in range(self.n_rows):
            first_index = self.indptr[i_row]
            last_index = self.indptr[i_row + 1]
            n_values = last_index - first_index

            n_compl_values = self.n_cols - n_values
            c_index = first_index
            c_value = self.indices[c_index]
            for i_col in range(self.n_cols):
                if i_col < c_value:
                    inv_array.indices[inv_indices_loc] = i_col
                    inv_indices_loc += 1
                elif c_index < last_index - 1:
                    c_index += 1
                    c_value = self.indices[c_index]
                else:
                    c_value = self.n_cols
            inv_array.indptr[i_row + 1] = inv_indices_loc

        return inv_array

    def __dealloc__(self):
        PyMem_Free(self.indptr)
        PyMem_Free(self.indices)


cdef class CSRBinaryMatrixRowSampler:
    cpdef vector[int32_t] draw(self, CSRBinaryMatrix mat, int32_t i_row, int32_t n_samples) except *:
        raise NotImplementedError


cdef class CSRBinaryMatrixRowSamplerWithReplacement(CSRBinaryMatrixRowSampler):
    cpdef vector[int32_t] draw(self, CSRBinaryMatrix mat, int32_t i_row, int32_t n_samples) except *:
        cdef vector[int32_t] samples_vec
        cdef int32_t first_index = mat.indptr[i_row]
        cdef int32_t n_values = mat.indptr[i_row + 1] - first_index
        cdef Py_ssize_t i
        for i in range(n_samples):
            samples_vec.push_back(mat.indices[first_index + randint(0, n_values - 1)])
        return samples_vec


cdef class CSRBinaryMatrixRowSamplerWithoutReplacement(CSRBinaryMatrixRowSampler):
    cpdef vector[int32_t] draw(self, CSRBinaryMatrix mat, int32_t i_row, int32_t n_samples) except *:
        cdef unordered_set[int32_t] samples_set
        cdef int32_t first_index = mat.indptr[i_row]
        cdef int32_t n_values = mat.indptr[i_row + 1] - first_index
        cdef int32_t i, pos, item
        for i in range(n_values - n_samples, n_values):
            pos = randint(0, i)
            item = mat.indices[first_index + pos]
            if samples_set.find(item) != samples_set.end():
                samples_set.insert(mat.indices[first_index + i])
            else:
                samples_set.insert(item)
        cdef vector[int32_t] samples_vec
        samples_vec.insert(samples_vec.end(), samples_set.begin(), samples_set.end())
        return samples_vec


cdef class SingleCellFingerprintStratifiedSampler:
    # (n_gene_groups, n_genes)
    cdef CSRBinaryMatrix gene_groups_csr
    
    # (n_genes, n_cells)
    cdef CSRBinaryMatrix expressing_cells_csr
    
    # (n_genes, n_cells)
    cdef CSRBinaryMatrix silent_cells_csr

    cdef int32_t n_genes
    cdef int32_t n_cells
    cdef int32_t n_gene_groups

    cdef int32_t genes_per_gene_group
    cdef int32_t expressing_cells_per_gene
    cdef int32_t silent_cells_per_gene
    
    cdef CSRBinaryMatrixRowSampler csr_row_sampler
    
    def __cinit__(
            self,
            CSRBinaryMatrix gene_groups_csr,
            CSRBinaryMatrix expressing_cells_csr,
            str sampling_strategy):
        
        self.n_gene_groups = gene_groups_csr.n_rows
        self.n_genes = expressing_cells_csr.n_rows
        self.n_cells = expressing_cells_csr.n_cols
        assert gene_groups_csr.n_cols == self.n_genes, \
            (f"The column dim of gene_groups_csr ({gene_groups_csr.n_cols}) must "
             f"match the row dim of expressing_cells_csr ({expressing_cells_csr.n_rows})")

        assert sampling_strategy in {'with_replacement', 'without_replacement'}
        if sampling_strategy == 'with_replacement':
            self.csr_row_sampler = CSRBinaryMatrixRowSamplerWithReplacement()
        elif sampling_strategy == 'without_replacement':
            self.csr_row_sampler = CSRBinaryMatrixRowSamplerWithoutReplacement()
        
        self.gene_groups_csr = gene_groups_csr
        self.expressing_cells_csr = expressing_cells_csr
        self.silent_cells_csr = ~expressing_cells_csr
        
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cpdef int32_t draw(self,
                        int32_t genes_per_gene_group,
                        int32_t expressing_cells_per_gene,
                        int32_t silent_cells_per_gene,
                        int32_t[::1] gene_index_memview,
                        int32_t[::1] cell_index_memview,
                        FLOAT_DTYPE[::1] gene_sampling_site_scale_factor_memview,
                        FLOAT_DTYPE[::1] cell_sampling_site_scale_factor_memview):

        cdef int32_t i_gene_group, i_gene, c_n_genes, c_gene_group_sz
        cdef FLOAT_DTYPE c_gene_scale_factor

        cdef int32_t c_expressing_cells_sz, c_n_expressing_cells
        cdef FLOAT_DTYPE c_expressing_cell_scale_factor
                
        cdef int32_t c_silent_cells_sz, c_n_silent_cells
        cdef FLOAT_DTYPE c_silent_cell_scale_factor

        cdef int32_t c_total_cells_for_gene
        cdef FLOAT_DTYPE c_fractionalized_gene_scale_factor
        
        cdef vector[int32_t] c_gene_indices
        cdef vector[int32_t].iterator i_gene_it

        cdef vector[int32_t] c_expressing_cell_indices
        cdef vector[int32_t].iterator i_expressing_cell_it

        cdef vector[int32_t] c_silent_cell_indices
        cdef vector[int32_t].iterator i_silent_cell_it
        
        cdef Py_ssize_t cell_ptr = 0
        cdef Py_ssize_t gene_ptr = 0
        cdef Py_ssize_t i
        
        # select genes
        for i_gene_group in range(self.n_gene_groups):
            
            # number of genes to draw from the gene group
            c_gene_group_sz = self.gene_groups_csr.get_non_zero_cols(i_gene_group)
            c_n_genes = min(genes_per_gene_group, c_gene_group_sz)
            
            if c_n_genes > 0:

                c_gene_indices = self.csr_row_sampler.draw(
                    self.gene_groups_csr, i_gene_group, c_n_genes)

                # weight of randomly drawn genes
                c_gene_scale_factor = (<FLOAT_DTYPE> c_gene_group_sz) / c_n_genes
                
                # select silent and expressing cells from each gene
                i_gene_it = c_gene_indices.begin()
                while i_gene_it != c_gene_indices.end():
                    i_gene = deref(i_gene_it)
                    inc(i_gene_it)

                    # draw expressing cells from "i_gene"
                    c_expressing_cells_sz = self.expressing_cells_csr.get_non_zero_cols(i_gene)
                    c_n_expressing_cells = min(expressing_cells_per_gene, c_expressing_cells_sz)

                    if c_n_expressing_cells > 0:

                        c_expressing_cell_indices = self.csr_row_sampler.draw(
                            self.expressing_cells_csr, i_gene, c_n_expressing_cells)

                        c_expressing_cell_scale_factor = (
                            (c_gene_scale_factor * c_expressing_cells_sz) / c_n_expressing_cells)

                        i_expressing_cell_it = c_expressing_cell_indices.begin()
                        while i_expressing_cell_it != c_expressing_cell_indices.end():
                            cell_index_memview[cell_ptr] = deref(i_expressing_cell_it)
                            cell_sampling_site_scale_factor_memview[cell_ptr] = \
                                c_expressing_cell_scale_factor
                            inc(i_expressing_cell_it)
                            cell_ptr += 1

                    # draw silent cells from "i_gene"
                    c_silent_cells_sz = self.silent_cells_csr.get_non_zero_cols(i_gene)
                    c_n_silent_cells = min(silent_cells_per_gene, c_silent_cells_sz)

                    if c_n_silent_cells > 0:

                        c_silent_cell_indices = self.csr_row_sampler.draw(
                            self.silent_cells_csr, i_gene, c_n_silent_cells)

                        c_silent_cell_scale_factor = (
                            (c_gene_scale_factor * c_silent_cells_sz) / c_n_silent_cells)

                        i_silent_cell_it = c_silent_cell_indices.begin()
                        while i_silent_cell_it != c_silent_cell_indices.end():
                            cell_index_memview[cell_ptr] = deref(i_silent_cell_it)
                            cell_sampling_site_scale_factor_memview[cell_ptr] = \
                                c_silent_cell_scale_factor
                            inc(i_silent_cell_it)
                            cell_ptr += 1

                    # gene sampling site effective ("fractionalized") scale factor
                    c_total_cells_for_gene = c_n_expressing_cells + c_n_silent_cells

                    if c_total_cells_for_gene > 0:

                        c_fractionalized_gene_scale_factor = \
                            c_gene_scale_factor / c_total_cells_for_gene

                        for i in range(c_total_cells_for_gene):
                            gene_index_memview[gene_ptr + i] = i_gene
                            gene_sampling_site_scale_factor_memview[gene_ptr + i] = \
                                c_fractionalized_gene_scale_factor

                        gene_ptr += c_total_cells_for_gene

        return cell_ptr
