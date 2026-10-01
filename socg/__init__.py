"""TANGRAM — Stateful Operator Coarse Graining (package namespace: socg).

Unified model U(R, s, a): geometry (BAOAB), conformational states
(Metropolis flips) and amino-acid identity (MH sequence moves with learned
proposals) as coupled variables. Peptide POC: Ala10 / CLN025.
Units across the project: length nm, time ps, energy kJ/mol,
force kJ/mol/nm, mass amu, temperature K. Angles are stored in
degrees in data files and handled in radians inside torch models.
"""

__version__ = "0.2.0"
