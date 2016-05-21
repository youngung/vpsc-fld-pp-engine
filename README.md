# VPSC-FLD-PP-ENGINE
Essential python scripts incorporated in VPSC-FLD to enable multi-threaded and
highly efficient computations to *predict* forming limit diagram (FLD) on the basis of
ViscoPlastic Self-Consistent (VPSC) crystal plasticity code developed by R. Lebensohn and C. Tome.

Note that VPSC-FLD is a separate repository but maintained in a private sector of USNISTGOV account in GitHub.
The essential portion of VPSC-FLD is publicly stored in this repository
for those who want to have a quick look at how Marciniak-Kuczynski model was incorporated into VPSC and how the multi-threaded
computation for VPSC-FLD was realized using Python's multiprocessing package.


# Features
- Considering the crystallographic texture and micro-mechanical constitutive models (as employed in VPSC)
  conduct forming limit 'virtual' tests on the basis of Marciniak-Kuczynski model.
- This requires a set of virtual tests under various conditions thus requires a number of
  VPSC runs.

The result can be summarized by the below figures, which is auto-generated by VPSC-FLD
![image of VPSC-FLD for an aluminum]
(https://github.com/youngung/vpsc-fld-pp-engine/blob/dev/images/vpsc-fld-ex01.png)


Technical papers for reference:
-------------------------------
1. A comparative study between micro- and macro-mechanical constitutive
 models developed for complex loading scenarios, **Y. Jeong**, F. Barlat,
 C. Tome, W. Wen (Submitted to International Journal of Plasticity)
2. Advances in Constitutive Modelling of Plasticity for forming
 Applications, F. Barlat, **Y. Jeong**, J. Ha, C. Tome, M-G. Lee,
 W. Wen, (submitted) AEPA 2016
3. Forming limit predictions using a self-consistent crystal plasticity
 framework: a case study for body-centered cubic materials, **Y. Jeong**,
 M-S. Pham, M. Iadicola, A. Creuziger, T. Foecke, Modelling
 and Simulation in Materials Science and Engineering 24 (5), 2016
4. Validation of Homogeneous Anisotropic Hardening Approach Based on
 Crystal Plasticity, **Y. Jeong**, F. Barlat, C. Tome, W. Wen (Accepted)
 ESAFORM 2016
5. Multiaxial constitutive behavior of an interstitial-free steel:
 measurements through X-ray and digital image correlation, **Y. Jeong**,
 T. Gnaeupel-Herold, M. Iadicola, A. Creuziger, Acta Materialia 112,
 84-93 (2016)
6. Texture-based forming limit prediction for Mg sheet alloys ZE10 and
 AZ31, Dirk Steglich, **Y. Jeong** (submitted to International Journal
  of Mechanical Sciences)
7. Forming limit diagram predictions using a self-consistent crystal
 plasticity model: a parametric study, **Y. Jeong**, M-S. Pham,
 M. Iadicola, A. Creuziger, Key Engineering Materials 651,
 193-198 (2015)



Do you want a copy of VPSC-FLD?
-------------------------------
This repository is not complete since VPSC-FLD requires VPSC source code.
For those who would like to have an access or to have a copy of the full
VPSC-FLD code, please contact me via youngung.jeong@gmail.com
