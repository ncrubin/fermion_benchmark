# Quantum Processor Sanity Testing with Non-Interacting Fermions

Prior to running any quantum algorithm one should validate the 
performance of the quantum device being used. Usually, a 
service provider will provide their own benchmarks reporting
overall device quality or qubit-by-qubit performance.  Due to device 
performance drift or calibration routine sensitivity these service
provider benchmarks can be poorly correlated with circuit level
performance for an algorithm run by a user.  An example of this
is cross-entropy benchmarking a gate in the XX-YY family.  Local
Z-errors commute through to the end of the XEB circuit and are
not visible.  Furthermore, randomization in the XEB circuit does 
not refocus the Z-coherent errors in XX-YY + local Z non-microwave
circuits. Therefore, it is important the user validates the provided
metrics for circuits that are close to the algorithm they are going
to run.  

In many chemistry and condensed-matter simulation experiments 
basis rotations are a ubiquitous algorithmic primitive.  Using a 
quantum processor with a gate in the XX-YY family the Givens rotation
gate can be implemented with two of these entangling gates and local 
Z-gates--i.e. no microwave is needed.  The circuits are efficiently 
simulable because though the simulation is for fermions, on a 1D chain
these circuits are equivalent to match-gate circuits.  

[Citations to recent work fermion Givens RB]

# Benchmark goals

In this set of python scripts we provide self-contained code to generate
random non-interacting fermion circuits in sqrt(iSWAP) gate sets, 
measurement code for analyzing post-selection and readout corrections,
and functions for extracting fidelity lower-bounds.  We also provide 
a simple visual benchmark for validating gross errors in circuit compilation
or qubits. The benchmark does not try to use the absolute best compilation 
for basis rotation circuits but instead uses the general circuit construction
from Clements [citation].

# Code standards

This benchmark code is self contained.  The functionality for generating circuits
analysis and plotting reproduced from OpenFermion and Recirq.  The reason for 
the code duplication is to have minimal dependencies to ensure the benchmark
is usable (from a code perspective) for as long as possible.  The code only relies
on common numpy, scipy, and pandas functions along with a stable release of Cirq.

# Outline

1. [main.py] The main interface to the code.  (In the future we should provide
   a nice argparse interface so people can run this from the command line.) This
   file launches the becnhmark and saves the data.  User input will be qubits to 
   benchmark.  Circuit depth is dictated by the number of qubits.
   
2. [circuits.py] Generate the givens rotation circuits and provide an interface to 
   measurement circuits if readout correction is being used.
   
3. [analysis.py] All functionality for analyzing the data: fidelity, plotting,
   eigen spectra (if fast fowarding), purity, etc.
   
4. [util.py] Utility module needed for circuit generation
   

   