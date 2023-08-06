"""Constant numbers used in remove-background."""

# Seed for random number generators.
RANDOM_SEED = 1234

# Factor by which the mode UMI count of the empty droplet plateau is
# multiplied to come up with a UMI cutoff below which no barcodes are used.
EMPIRICAL_LOW_UMI_TO_EMPTY_DROPLET_THRESHOLD = 0.5

# Features with fewer than this many counts summed over all empty droplets
# are ignored during analysis, and the output count matrix for these features
# is identical to the input.
COUNTS_IN_EMPTIES_LOW_LIMIT = 0
AMBIENT_COUNTS_IN_CELLS_LOW_LIMIT = 0.1

# Default prior for the standard deviation of the LogNormal distribution for
# cell size, used only in the case of the 'simple' model.
SIMPLE_MODEL_D_STD_PRIOR = 0.2

# Probability cutoff for determining which droplets contain cells and which
# are empty.  The droplets n with inferred probability q_n > CELL_PROB_CUTOFF
# are saved as cells in the filtered output file.
CELL_PROB_CUTOFF = 0.5

# Default UMI cutoff. Droplets with UMI less than this will be completely
# excluded from the analysis.
LOW_UMI_CUTOFF = 5

# Default total number of droplets to include in the analysis.
TOTAL_DROPLET_DEFAULT = 25000

# Fraction of the data used for training (versus testing).
TRAINING_FRACTION = 0.9

# Size of minibatch by default.
DEFAULT_BATCH_SIZE = 128

# Fraction of totally empty droplets that makes up each minibatch, by default.
FRACTION_EMPTIES = 0.2

# Prior on rho, the swapping fraction: the two concentration parameters alpha and beta.
RHO_ALPHA_PRIOR = 1.5
RHO_BETA_PRIOR = 50.

# Constraints on rho posterior latents.
RHO_PARAM_MIN = 0.001
RHO_PARAM_MAX = 1000.

# Prior on epsilon, the RT efficiency concentration parameter [Gamma(alpha, alpha)].
EPSILON_PRIOR = 50.

# Prior used for the global overdispersion parameter.
PHI_LOC_PRIOR = 0.2
PHI_SCALE_PRIOR = 0.2

# Prior for mixture weights in Gaussian mixture model.
GMM_ALPHA_PRIOR = 1e-1
GMM_COMPONENTS = 10
GMM_EPOCHS = 500

# Initial value of global latent scale for d_cell.
D_CELL_SCALE_INIT = 0.02

# Scale used to regularize values of logit cell probability (mean zero).
P_LOGIT_SCALE = 2.

# False to use an approximate log_prob computation which is much faster.
USE_EXACT_LOG_PROB = False

# If using an exact log_prob computation, we integrate numerically over this size range.
NBPC_EXACT_N_TERMS = 50

# Negative binomial poisson convolution likelihood calculation: numerical safeguards.
NBPC_MU_EPS_SAFEGAURD = 1e-10
NBPC_ALPHA_EPS_SAFEGAURD = 1e-10
NBPC_LAM_EPS_SAFEGAURD = 1e-10
POISSON_EPS_SAFEGAURD = 1e-10

# Scale factors for loss function regularization terms: semi-supervision.
REG_SCALE_AMBIENT_EXPRESSION = 0.01
REG_SCALE_SOFT_SUPERVISION = 10.0

# Regularize logit probabilities toward this lognormal distribution.
REG_LOGIT_MEAN = 10.0
REG_LOGIT_SCALE = 0.2
REG_LOGIT_SOFT_SCALE = 1.0

# Number of cells used to esitmate posterior regularization lambda. Memory hungry.
CELLS_POSTERIOR_REG_CALC = 100

# Posterior regularization constant's upper and lower bounds.
POSTERIOR_REG_MIN = 0.01
POSTERIOR_REG_MAX = 500
POSTERIOR_REG_SEARCH_MAX_ITER = 50

# For AnnData h5ad files, fewer than this many barcodes will trigger a warning,
# since it indicates that the file might be "filtered" to cells only.
MINIMUM_BARCODES_H5AD = 1e5

# Batch size for posterior inference.
PROB_POSTERIOR_BATCH_SIZE = 128

# Name of checkpoint file.
CHECKPOINT_FILE_NAME = 'ckpt.tar.gz'

# Whether to create an extended report (for development purposes).
EXTENDED_REPORT = False

# Maximum batch size
MAX_BATCH_SIZE = 256
SMALLEST_ALLOWED_BATCH = 4  # BatchNorm chokes if there is only 1 cell in last batch

# Guesses during prior estimation
MAX_TOTAL_DROPLETS_GUESSED = 70000
MAX_EMPTIES_TO_INCLUDE = 20000
NUM_EMPTIES_INCREMENT = 20000  # if input expected_cells > heuristic prior total_drops
