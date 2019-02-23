# CellBender

_CellBender_ is the project codename for single-cell data analysis methods developed at Data Science and Data Engineering (DSDE), Methods Division, Broad Institute.

# Mandate

To fix the _ad hoc analysis crisis_, also known as the reproducibility crisis, in the single-cell transcriptomics domain by means of _developing principled probabilistic solutions to existing and novel questions_.

# Manifesto

- We use **logic** and **rational reasoning**, and we avoid _argument by popularity_, _argument by authority_, and _ad hoc fallacies_:

  * Examples of **bad arguments**:

    - We got more _sensible_ results when we adopted method X. So, we dropped method Y in favor of method X.
    
    - It is a _common practice in the community_ to use method X for data normalization and cleanup.
  	
  	- Person _X_ advocates for method _Y_ and has found _good_ results with it.

  * Examples of **good arguments**:

  	- Assuming _X_, _Y_, _Z_, we infer _a_ to be in range _[a_0, a_1]_ with 90% probability. Provided that our assumptions are experimentally valid, our results provide support for model _M_ with probability _p_. This finding is consistent with the analysis of Ref. _W_, which is fully rigorous. This finding is inconsistent with the analysis of Ref. _R_, but that analysis is _ad hoc_.

  	- Method _X_ for data normalization and cleanup is justified under the following assumptions: _Y_, _Z_, _W_. We show irrefutable experimental evidence for the validity of all three assumptions, and therefore, we adopt method _X_.

  	- Person _X_ advocates for method _Y_, which is based on sound probabilistic modeling with clearly defined assumptions. Moreover, the assumptions underlying method _Y_ have strong logical and empirical underpinning. Therefore, we also adopt method _Y_. 

- We use mathematical terms _precisely_, and _profusely_. We do not sacrifice _clarity_.

- We _never_ report point estimates without credible intervals, and we _never_ propagate point estimates in pipelines for convenience or speed.

- All _tools_ must be accompanied by an up-to-date detailed _LaTeX_ document that explain all assumptions, approximations and derivations in detail. **Nothing will be swept under the rug.**

- All methods must be supported by numerical simulations (for consistency check), and by existing experimental data (for relevance), or with experimental proposals (for validation).

- In the spirit of openness and collaboration, all technical documents and developments, including work-in-progress and unpublished work, are publicly visible. _Found an interesting idea here? great! please write to us and we will gladly join efforts._

# Contributors

- Mehrtash Babadi: _Harvard University, Department of Physics, PhD_
- Stephen Fleming: _Harvard University, Department of Physics, PhD_
- Luca D'Alessio: _Boston University, Department of Physics, PhD_

