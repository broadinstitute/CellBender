import unittest
import os
import warnings

import cellbender
import cellbender.remove_background.model
from cellbender.remove_background.train import run_inference
from cellbender.remove_background.data.extras.simulate import \
    simulate_dataset_with_ambient_rna
from cellbender.remove_background.data.dataset import \
    SingleCellRNACountsDataset, write_matrix_to_cellranger_h5, \
    get_matrix_from_cellranger_h5
import numpy as np
import sys


class TestConsole(unittest.TestCase):

    def test_data_simulation_and_write_and_read(self):
        """Run basic tests of data simulation and read and write functionality.

        Test the ability to read/write data to/from an HDF5 file using pytables.
        This necessitates creating a small simulated dataset, writing it to
        a temporary file, reading the file back in, and deleting the
        temporary file.

        """

        try:

            # This is here to suppress the numpy warning triggered by
            # scipy.sparse.
            warnings.simplefilter("ignore")

            # Generate a simulated dataset with ambient RNA.
            n_cells = 100
            csr_barcode_gene_synthetic, _, chi, _ = \
                simulate_dataset_with_ambient_rna(n_cells=n_cells,
                                                  n_empty=3 * n_cells,
                                                  clusters=1, n_genes=1000,
                                                  d_cell=2000, d_empty=100,
                                                  ambient_different=False)

            # Generate some names for genes and barcodes.
            gene_names = np.array([f'g_{i}' for i in
                                   range(csr_barcode_gene_synthetic.shape[1])])
            barcode_names = \
                np.array([f'bc_{i}' for i in
                          range(csr_barcode_gene_synthetic.shape[0])])

            # Save the data to a temporary file.
            temp_file_name = 'testfile.h5'
            write_matrix_to_cellranger_h5(temp_file_name,
                                          loss=None,
                                          gene_names=gene_names,
                                          barcodes=barcode_names,
                                          inferred_count_matrix=
                                          csr_barcode_gene_synthetic.tocsc(),
                                          ambient_expression=chi[0, :])

            # Read the data back in.
            reconstructed = get_matrix_from_cellranger_h5(temp_file_name)
            new_matrix = reconstructed['matrix']

            # Check that the data matches.
            assert (csr_barcode_gene_synthetic.sum(axis=1) ==
                    new_matrix.sum(axis=1)).all(), \
                "Saved and re-opened data is not accurate."
            assert (csr_barcode_gene_synthetic.sum(axis=0) ==
                    new_matrix.sum(axis=0)).all(), \
                "Saved and re-opened data is not accurate."

            # Remove the temporary file.
            os.remove(temp_file_name)

            return 1

        except TestConsole.failureException:

            return 0

    def test_inference(self):
        """Run a basic tests doing inference on a synthetic dataset.

        Runs the inference procedure on CPU.

        """

        try:

            n_cells = 100

            # Generate a simulated dataset with ambient RNA.
            csr_barcode_gene_synthetic, _, _, _ = \
                simulate_dataset_with_ambient_rna(n_cells=n_cells,
                                                  n_empty=3 * n_cells,
                                                  clusters=1, n_genes=1000,
                                                  d_cell=2000, d_empty=100,
                                                  ambient_different=False)

            # Fake some parsed command line inputs.
            args = ObjectWithAttributes()
            args.use_cuda = False
            args.z_hidden_dims = [100]
            args.d_hidden_dims = [10, 2]
            args.p_hidden_dims = [100, 10]
            args.z_dim = 10
            args.learning_rate = 0.001
            args.epochs = 10
            args.model = "full"
            args.fraction_empties = 0.5
            args.use_jit = True
            args.training_fraction = 0.9

            args.expected_cell_count = n_cells

            # Wrap simulated count matrix in a Dataset object.
            dataset_obj = SingleCellRNACountsDataset()
            dataset_obj.data = \
                {'matrix': csr_barcode_gene_synthetic,
                 'gene_names':
                     np.array([f'g{n}' for n in
                               range(csr_barcode_gene_synthetic.shape[1])]),
                 'barcodes':
                     np.array([f'bc{n}' for n in
                               range(csr_barcode_gene_synthetic.shape[0])])}
            dataset_obj.priors['n_cells'] = n_cells
            dataset_obj._trim_dataset_for_analysis()
            dataset_obj._estimate_priors()

            # Run inference on this simulated dataset.
            inferred_model = run_inference(dataset_obj, args)

            # Get encodings from the trained model.
            z, d, p = cellbender.remove_background.model.\
                get_encodings(inferred_model, dataset_obj)

            # Make the background-subtracted dataset.
            inferred_count_matrix = cellbender.remove_background.model.\
                generate_maximum_a_posteriori_count_matrix(z, d, p,
                                                           inferred_model,
                                                           dataset_obj)

            # Get the inferred background RNA expression from the model.
            ambient_expression = cellbender.remove_background.model.\
                get_ambient_expression_from_pyro_param_store()

            return 1

        except TestConsole.failureException:

            return 0


class ObjectWithAttributes(object):
    """Exists only to populate the args data structure with attributes."""
    pass


def main():
    """Run unit tests."""

    tester = TestConsole()

    passed_tests = 0

    passed_tests += tester.test_data_simulation_and_write_and_read()
    passed_tests += tester.test_inference()

    sys.stdout.write(f'Passed {passed_tests} of 2 tests.\n\n')
