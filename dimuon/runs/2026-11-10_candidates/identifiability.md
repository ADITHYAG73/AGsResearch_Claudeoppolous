# Identifiability of candidate first-step rules (synthetic)

Confusable pair (T, C) at noise eps: noise-free held-out misfit m(T,C) < 1.118*eps (equivalently err_C/err_T < 1.5 under i.i.d. noise on D). Fixed candidates only; families are reported as parameter recovery.


## generator = parametric


### eps = 1e-06: fixed candidates confusable with the true rule

| true rule | r=1 | r=2 | r=4 | r=8 | r=16 | r=64 |
|---|---|---|---|---|---|---|
| frob | cpolar_colwhite, col_norm, ns_quintic3, polar, ns_cubic5, ns_cubic15, ns_quintic5, spec_p0.5, ns_quintic8, ns_quintic5_f32 | - | - | - | - | - |
| sign | cpolar_rowwhite, row_norm | - | - | - | - | - |
| row_norm | cpolar_rowwhite, sign | - | - | - | - | - |
| col_norm | ns_quintic3, polar, ns_cubic5, ns_cubic15, cpolar_colwhite, spec_p0.5, ns_quintic5, ns_quintic8, frob, ns_quintic5_f32 | - | - | - | - | - |
| polar | frob, col_norm, ns_cubic5, ns_cubic15, ns_quintic8, cpolar_colwhite, ns_quintic5, ns_quintic3, spec_p0.5, ns_quintic5_f32 | ns_cubic15 | ns_cubic15 | - | - | - |
| ns_quintic5 | spec_p0.5, ns_quintic3, frob, col_norm, ns_quintic8, cpolar_colwhite, polar, ns_cubic5, ns_cubic15, ns_quintic5_f32 | ns_quintic5_f32 | ns_quintic5_f32 | ns_quintic5_f32 | ns_quintic5_f32 | - |
| ns_quintic3 | ns_quintic8, cpolar_colwhite, frob, col_norm, polar, ns_cubic5, ns_cubic15, ns_quintic5, spec_p0.5, ns_quintic5_f32 | - | - | - | - | - |
| ns_quintic8 | spec_p0.5, ns_quintic3, ns_quintic5, col_norm, frob, polar, ns_cubic5, ns_cubic15, cpolar_colwhite, ns_quintic5_f32 | - | - | - | - | - |
| ns_cubic5 | frob, col_norm, ns_quintic8, polar, ns_cubic15, ns_quintic3, cpolar_colwhite, ns_quintic5, spec_p0.5, ns_quintic5_f32 | - | - | - | - | - |
| ns_cubic15 | polar, ns_cubic5, ns_quintic5, frob, col_norm, ns_quintic8, spec_p0.5, cpolar_colwhite, ns_quintic3, ns_quintic5_f32 | - | - | - | - | - |
| spec_p0.5 | ns_quintic3, cpolar_colwhite, ns_quintic5, polar, ns_cubic5, ns_cubic15, frob, col_norm, ns_quintic8, ns_quintic5_f32 | - | - | - | - | - |
| dpre_-.25_-.25 | cpolar_dshampoo | - | - | - | - | - |
| cpolar_rowwhite | sign, row_norm | - | - | - | - | - |
| cpolar_colwhite | frob, col_norm, spec_p0.5, ns_quintic8, ns_quintic5, polar, ns_cubic5, ns_cubic15, ns_quintic3, ns_quintic5_f32 | - | - | - | - | - |
| cpolar_dshampoo | dpre_-.25_-.25 | - | - | - | - | - |
| ns_quintic5_f32 | spec_p0.5, ns_quintic3, ns_quintic5, col_norm, polar, ns_cubic5, ns_cubic15, ns_quintic8, frob, cpolar_colwhite | - | ns_quintic5 | ns_quintic5 | ns_quintic5 | - |
| ns_quintic5_bf16 | - | - | - | - | - | - |
| cpolar(.3,-.4,-.2,.1) | - | - | - | - | - | - |

### eps = 0.001: fixed candidates confusable with the true rule

| true rule | r=1 | r=2 | r=4 | r=8 | r=16 | r=64 |
|---|---|---|---|---|---|---|
| frob | cpolar_colwhite, col_norm, ns_quintic3, polar, ns_cubic5, ns_cubic15, ns_quintic5, spec_p0.5, ns_quintic8, ns_quintic5_f32 | - | - | - | - | - |
| sign | cpolar_rowwhite, row_norm | - | - | - | - | - |
| row_norm | cpolar_rowwhite, sign | - | - | - | - | - |
| col_norm | ns_quintic3, polar, ns_cubic5, ns_cubic15, cpolar_colwhite, spec_p0.5, ns_quintic5, ns_quintic8, frob, ns_quintic5_f32 | - | - | - | - | - |
| polar | frob, col_norm, ns_cubic5, ns_cubic15, ns_quintic8, cpolar_colwhite, ns_quintic5, ns_quintic3, spec_p0.5, ns_quintic5_f32 | ns_cubic15 | ns_cubic15 | ns_cubic15 | ns_cubic15 | - |
| ns_quintic5 | spec_p0.5, ns_quintic3, frob, col_norm, ns_quintic8, cpolar_colwhite, polar, ns_cubic5, ns_cubic15, ns_quintic5_f32 | ns_quintic5_f32 | ns_quintic5_f32 | ns_quintic5_f32 | ns_quintic5_f32 | ns_quintic5_f32 |
| ns_quintic3 | ns_quintic8, cpolar_colwhite, frob, col_norm, polar, ns_cubic5, ns_cubic15, ns_quintic5, spec_p0.5, ns_quintic5_f32 | - | - | - | - | - |
| ns_quintic8 | spec_p0.5, ns_quintic3, ns_quintic5, col_norm, frob, polar, ns_cubic5, ns_cubic15, cpolar_colwhite, ns_quintic5_f32 | - | - | - | - | - |
| ns_cubic5 | frob, col_norm, ns_quintic8, polar, ns_cubic15, ns_quintic3, cpolar_colwhite, ns_quintic5, spec_p0.5, ns_quintic5_f32 | - | - | - | - | - |
| ns_cubic15 | polar, ns_cubic5, ns_quintic5, frob, col_norm, ns_quintic8, spec_p0.5, cpolar_colwhite, ns_quintic3, ns_quintic5_f32 | - | polar | polar | - | - |
| spec_p0.5 | ns_quintic3, cpolar_colwhite, ns_quintic5, polar, ns_cubic5, ns_cubic15, frob, col_norm, ns_quintic8, ns_quintic5_f32 | - | - | - | - | - |
| dpre_-.25_-.25 | cpolar_dshampoo | - | - | - | - | - |
| cpolar_rowwhite | sign, row_norm | - | - | - | - | - |
| cpolar_colwhite | frob, col_norm, spec_p0.5, ns_quintic8, ns_quintic5, polar, ns_cubic5, ns_cubic15, ns_quintic3, ns_quintic5_f32 | - | - | - | - | - |
| cpolar_dshampoo | dpre_-.25_-.25 | - | - | - | - | - |
| ns_quintic5_f32 | spec_p0.5, ns_quintic3, ns_quintic5, col_norm, polar, ns_cubic5, ns_cubic15, ns_quintic8, frob, cpolar_colwhite | ns_quintic5 | ns_quintic5 | ns_quintic5 | ns_quintic5 | ns_quintic5 |
| ns_quintic5_bf16 | - | - | - | - | - | - |
| cpolar(.3,-.4,-.2,.1) | - | - | - | - | - | - |

### eps = 0.01: fixed candidates confusable with the true rule

| true rule | r=1 | r=2 | r=4 | r=8 | r=16 | r=64 |
|---|---|---|---|---|---|---|
| frob | cpolar_colwhite, col_norm, ns_quintic3, polar, ns_cubic5, ns_cubic15, ns_quintic5, spec_p0.5, ns_quintic8, ns_quintic5_f32 | - | - | - | - | - |
| sign | cpolar_rowwhite, row_norm | - | - | - | - | - |
| row_norm | cpolar_rowwhite, sign | - | - | - | - | - |
| col_norm | ns_quintic3, polar, ns_cubic5, ns_cubic15, cpolar_colwhite, spec_p0.5, ns_quintic5, ns_quintic8, frob, ns_quintic5_f32 | - | - | - | - | - |
| polar | frob, col_norm, ns_cubic5, ns_cubic15, ns_quintic8, cpolar_colwhite, ns_quintic5, ns_quintic3, spec_p0.5, ns_quintic5_f32 | ns_cubic15 | ns_cubic15 | ns_cubic15 | ns_cubic15 | - |
| ns_quintic5 | spec_p0.5, ns_quintic3, frob, col_norm, ns_quintic8, cpolar_colwhite, polar, ns_cubic5, ns_cubic15, ns_quintic5_f32 | ns_quintic5_f32 | ns_quintic5_f32 | ns_quintic5_f32 | ns_quintic5_f32 | ns_quintic5_f32 |
| ns_quintic3 | ns_quintic8, cpolar_colwhite, frob, col_norm, polar, ns_cubic5, ns_cubic15, ns_quintic5, spec_p0.5, ns_quintic5_f32 | - | - | - | - | - |
| ns_quintic8 | spec_p0.5, ns_quintic3, ns_quintic5, col_norm, frob, polar, ns_cubic5, ns_cubic15, cpolar_colwhite, ns_quintic5_f32 | - | - | - | - | - |
| ns_cubic5 | frob, col_norm, ns_quintic8, polar, ns_cubic15, ns_quintic3, cpolar_colwhite, ns_quintic5, spec_p0.5, ns_quintic5_f32 | - | - | - | - | - |
| ns_cubic15 | polar, ns_cubic5, ns_quintic5, frob, col_norm, ns_quintic8, spec_p0.5, cpolar_colwhite, ns_quintic3, ns_quintic5_f32 | polar | polar | polar | polar | - |
| spec_p0.5 | ns_quintic3, cpolar_colwhite, ns_quintic5, polar, ns_cubic5, ns_cubic15, frob, col_norm, ns_quintic8, ns_quintic5_f32 | - | - | - | - | - |
| dpre_-.25_-.25 | cpolar_dshampoo | - | - | - | - | - |
| cpolar_rowwhite | sign, row_norm | - | - | - | - | - |
| cpolar_colwhite | frob, col_norm, spec_p0.5, ns_quintic8, ns_quintic5, polar, ns_cubic5, ns_cubic15, ns_quintic3, ns_quintic5_f32 | - | - | - | - | - |
| cpolar_dshampoo | dpre_-.25_-.25 | - | - | - | - | - |
| ns_quintic5_f32 | spec_p0.5, ns_quintic3, ns_quintic5, col_norm, polar, ns_cubic5, ns_cubic15, ns_quintic8, frob, cpolar_colwhite | ns_quintic5 | ns_quintic5 | ns_quintic5 | ns_quintic5 | ns_quintic5 |
| ns_quintic5_bf16 | - | - | - | - | - | - |
| cpolar(.3,-.4,-.2,.1) | - | - | - | - | - | - |

### noise-free misfit to the NEAREST other fixed candidate (log10)

| true rule | r=1 | r=2 | r=4 | r=8 | r=16 | r=64 |
|---|---|---|---|---|---|---|
| frob | -16.0 (cpolar_colwhite) | -0.8 (spec_p0.5) | -0.6 (spec_p0.5) | -0.5 (spec_p0.5) | -0.5 (spec_p0.5) | -0.5 (ns_cubic5) |
| sign | -15.2 (cpolar_rowwhite) | -0.4 (row_norm) | -0.3 (row_norm) | -0.2 (row_norm) | -0.2 (row_norm) | -0.2 (row_norm) |
| row_norm | -15.4 (cpolar_rowwhite) | -0.3 (sign) | -0.3 (sign) | -0.3 (dpre_-.25_-.25) | -0.3 (dpre_-.25_-.25) | -0.3 (dpre_-.25_-.25) |
| col_norm | -15.8 (ns_quintic3) | -0.4 (spec_p0.5) | -0.3 (spec_p0.5) | -0.4 (spec_p0.5) | -0.3 (spec_p0.5) | -0.3 (ns_cubic5) |
| polar | -15.9 (frob) | -10.1 (ns_cubic15) | -11.0 (ns_cubic15) | -4.0 (ns_cubic15) | -3.7 (ns_cubic15) | -1.7 (ns_cubic15) |
| ns_quintic5 | -16.0 (spec_p0.5) | -6.0 (ns_quintic5_f32) | -6.1 (ns_quintic5_f32) | -6.1 (ns_quintic5_f32) | -6.0 (ns_quintic5_f32) | -5.9 (ns_quintic5_f32) |
| ns_quintic3 | -15.8 (ns_quintic8) | -1.5 (ns_quintic5) | -1.3 (ns_quintic5_f32) | -0.8 (ns_quintic5_f32) | -0.6 (ns_quintic5_f32) | -0.5 (ns_quintic5_f32) |
| ns_quintic8 | -16.0 (spec_p0.5) | -0.9 (ns_cubic15) | -0.7 (ns_cubic15) | -0.8 (ns_cubic15) | -0.7 (polar) | -0.7 (ns_cubic15) |
| ns_cubic5 | -15.9 (frob) | -0.7 (spec_p0.5) | -0.5 (spec_p0.5) | -0.6 (spec_p0.5) | -0.7 (spec_p0.5) | -0.7 (spec_p0.5) |
| ns_cubic15 | -15.9 (ns_cubic5) | -2.7 (polar) | -3.5 (polar) | -4.6 (polar) | -2.9 (polar) | -1.8 (polar) |
| spec_p0.5 | -15.7 (ns_quintic3) | -0.8 (frob) | -0.6 (frob) | -0.5 (ns_cubic5) | -0.7 (ns_cubic5) | -0.7 (ns_cubic5) |
| dpre_-.25_-.25 | -15.7 (cpolar_dshampoo) | -0.2 (spec_p0.5) | -0.3 (spec_p0.5) | -0.3 (spec_p0.5) | -0.3 (spec_p0.5) | -0.3 (ns_cubic5) |
| cpolar_rowwhite | -15.5 (row_norm) | -0.3 (row_norm) | -0.3 (cpolar_dshampoo) | -0.3 (cpolar_dshampoo) | -0.3 (cpolar_dshampoo) | -0.3 (cpolar_dshampoo) |
| cpolar_colwhite | -15.8 (col_norm) | -0.4 (ns_cubic15) | -0.5 (polar) | -0.5 (polar) | -0.6 (ns_cubic15) | -0.6 (polar) |
| cpolar_dshampoo | -15.8 (dpre_-.25_-.25) | -0.2 (ns_cubic15) | -0.3 (ns_cubic15) | -0.3 (ns_cubic15) | -0.3 (ns_cubic15) | -0.4 (polar) |
| ns_quintic5_f32 | -6.5 (spec_p0.5) | -5.9 (ns_quintic5) | -6.1 (ns_quintic5) | -6.0 (ns_quintic5) | -6.0 (ns_quintic5) | -5.9 (ns_quintic5) |
| ns_quintic5_bf16 | -1.7 (ns_quintic3) | -1.2 (ns_quintic5_f32) | -1.3 (ns_quintic5) | -1.4 (ns_quintic5_f32) | -1.4 (ns_quintic5_f32) | -1.6 (ns_quintic5_f32) |
| cpolar(.3,-.4,-.2,.1) | -0.7 (ns_quintic5_bf16) | -0.6 (ns_cubic5) | -0.5 (spec_p0.5) | -0.5 (ns_cubic15) | -0.5 (ns_cubic15) | -0.5 (ns_cubic15) |

### noise-free family fits (fitted params; held-out misfit)

| true rule | r | FIT spec_power | FIT diag_precond | FIT cpolar | FIT ns5_coeffs |
|---|---|---|---|---|---|
| frob | 1 | (+1.100) 2.0e-16 | (+0.000,-0.945) 1.4e-16 | (-0.456,+0.456,-0.627,-0.623) 2.8e-16 | (+2.063,-1.183,+0.677) 1.4e-16 |
| frob | 2 | (+1.000) 2.0e-09 | (-0.000,+0.000) 1.9e-16 | (+0.710,-0.708,-0.590,+0.199) 6.1e-02 | (+0.627,-0.203,+0.283) 6.1e-08 |
| frob | 4 | (+1.000) 7.8e-09 | (+0.000,-0.000) 1.5e-16 | (+0.795,-0.783,-0.282,+0.165) 1.6e-01 | (+0.871,+0.000,-0.000) 7.2e-09 |
| frob | 8 | (+1.000) 2.1e-09 | (+0.000,+0.000) 4.9e-16 | (+0.909,-0.898,-0.098,+0.268) 3.4e-01 | (+0.676,+0.000,-0.000) 9.5e-09 |
| frob | 16 | (+1.000) 1.8e-09 | (-0.000,-0.000) 2.7e-16 | (+0.842,-0.829,+0.175,+0.459) 4.0e-01 | (+0.490,+0.000,+0.000) 2.3e-09 |
| frob | 64 | (+1.000) 2.2e-09 | (-0.000,+0.000) 5.7e-16 | (+0.378,-0.362,+0.552,+0.627) 4.8e-01 | (+0.676,-0.000,+0.000) 3.5e-09 |
| row_norm | 1 | (+0.390) 8.8e-01 | (-0.500,-0.625) 7.2e-16 | (-0.273,-0.227,+0.689,+0.073) 4.8e-16 | (+2.104,-0.271,+0.233) 8.8e-01 |
| row_norm | 2 | (+0.560) 8.7e-01 | (-0.500,-0.000) 2.8e-16 | (+0.321,-0.812,-0.801,+0.246) 1.3e-01 | (+0.769,-3.484,+3.645) 8.7e-01 |
| row_norm | 4 | (+0.622) 8.3e-01 | (-0.500,+0.000) 1.7e-16 | (+0.401,-0.890,-0.095,+0.201) 2.5e-01 | (+0.532,-1.639,+1.689) 8.3e-01 |
| row_norm | 8 | (+0.681) 8.4e-01 | (-0.500,+0.000) 1.6e-16 | (+0.423,-0.895,+0.065,+0.306) 4.5e-01 | (+1.234,-0.167,+0.033) 8.4e-01 |
| row_norm | 16 | (+0.739) 8.1e-01 | (-0.500,-0.000) 1.7e-16 | (+0.377,-0.857,+0.356,+0.465) 5.2e-01 | (+0.923,-0.382,+0.117) 8.1e-01 |
| row_norm | 64 | (+0.730) 7.8e-01 | (-0.500,+0.000) 1.8e-16 | (+0.105,-0.603,+0.590,+0.593) 5.7e-01 | (+1.038,-1.503,+3.782) 7.8e-01 |
| col_norm | 1 | (+0.415) 2.3e-16 | (+0.000,+0.600) 1.6e-16 | (-0.459,+0.459,-0.620,-0.626) 3.3e-16 | (+2.154,-4.050,+0.199) 2.9e-16 |
| col_norm | 2 | (+0.406) 3.9e-01 | (+0.000,-0.500) 2.2e-16 | (+0.698,-0.693,-0.275,-0.369) 7.6e-02 | (+4.185,-2.961,+0.554) 5.5e-01 |
| col_norm | 4 | (+0.575) 4.6e-01 | (-0.000,-0.500) 1.3e-16 | (+0.840,-0.830,-0.603,-0.292) 2.6e-01 | (+1.822,-0.074,+0.002) 4.6e-01 |
| col_norm | 8 | (+0.681) 3.9e-01 | (+0.000,-0.500) 2.4e-16 | (+0.890,-0.861,-0.588,-0.092) 4.1e-01 | (+1.774,-0.103,+0.006) 3.9e-01 |
| col_norm | 16 | (+0.624) 4.5e-01 | (+0.000,-0.500) 1.7e-16 | (+0.798,-0.772,-0.648,-0.065) 5.3e-01 | (+1.498,-0.205,+0.032) 4.5e-01 |
| col_norm | 64 | (+0.614) 4.6e-01 | (+0.000,-0.500) 1.5e-16 | (+0.307,-0.254,-0.525,-0.013) 6.7e-01 | (+1.032,-1.757,-0.112) 4.6e-01 |
| polar | 1 | (+1.114) 1.1e-16 | (-0.000,-0.885) 1.5e-16 | (-0.456,+0.456,-0.627,-0.104) 1.7e-16 | (+2.149,-7.060,+3.878) 1.4e-16 |
| polar | 2 | (-0.000) 9.0e-09 | (-0.019,-0.264) 6.7e-01 | (+0.000,-0.000,-0.000,-0.000) 9.0e-14 | (+2.687,-4.351,+2.644) 2.9e-02 |
| polar | 4 | (-0.000) 4.9e-09 | (-0.064,-0.373) 6.5e-01 | (+0.000,-0.000,-0.000,-0.000) 3.9e-14 | (+2.637,-3.602,+1.959) 1.2e-02 |
| polar | 8 | (+0.000) 1.0e-08 | (-0.062,-0.388) 7.2e-01 | (-0.000,+0.000,+0.000,+0.000) 1.1e-13 | (+2.626,-4.837,+3.494) 3.1e-02 |
| polar | 16 | (+0.000) 5.9e-09 | (-0.059,-0.365) 6.8e-01 | (-0.000,+0.000,+0.000,-0.000) 5.2e-14 | (+2.639,-7.513,+8.268) 6.8e-01 |
| polar | 64 | (+0.000) 6.2e-09 | (-0.096,-0.360) 7.2e-01 | (+0.000,-0.000,+0.000,-0.000) 1.0e-13 | (+2.931,-15.736,+30.232) 5.0e-01 |
| ns_quintic5 | 1 | (-0.277) 1.5e-16 | (+0.000,-0.931) 2.1e-16 | (-0.458,+0.458,-0.104,+0.414) 2.2e-16 | (+2.229,-4.508,+1.000) 1.3e-16 |
| ns_quintic5 | 2 | (-0.083) 9.4e-02 | (-0.053,-0.476) 7.6e-01 | (-0.089,+0.088,+0.003,-0.048) 7.8e-02 | (+3.444,-4.775,+2.032) 4.3e-09 |
| ns_quintic5 | 4 | (-0.036) 1.7e-01 | (-0.054,-0.514) 8.0e-01 | (-0.018,+0.018,-0.002,-0.059) 1.8e-01 | (+3.444,-4.775,+2.031) 7.2e-10 |
| ns_quintic5 | 8 | (-0.001) 1.7e-01 | (-0.060,-0.354) 6.1e-01 | (-0.000,+0.000,-0.000,+0.005) 1.7e-01 | (+3.444,-4.775,+2.032) 1.3e-08 |
| ns_quintic5 | 16 | (-0.001) 1.8e-01 | (-0.065,-0.332) 6.7e-01 | (+0.002,-0.002,+0.000,-0.003) 1.8e-01 | (+3.444,-4.775,+2.032) 4.4e-09 |
| ns_quintic5 | 64 | (-0.017) 1.7e-01 | (-0.103,-0.348) 7.4e-01 | (-0.000,+0.000,+0.000,-0.010) 1.7e-01 | (+3.445,-4.775,+2.031) 4.3e-09 |
| spec_p0.5 | 1 | (+1.128) 1.8e-16 | (-0.000,-0.150) 2.3e-16 | (-0.456,+0.456,-0.627,-0.104) 2.5e-16 | (+4.059,-5.727,+2.418) 2.4e-16 |
| spec_p0.5 | 2 | (+0.500) 5.1e-09 | (-0.002,-0.033) 1.7e-01 | (+0.498,-0.498,-0.009,+0.017) 2.3e-02 | (+0.209,-14.094,+13.996) 1.6e-01 |
| spec_p0.5 | 4 | (+0.500) 7.4e-09 | (-0.020,-0.150) 2.5e-01 | (+0.647,-0.650,+0.032,-0.016) 1.0e-01 | (+1.705,-0.490,+0.105) 5.9e-02 |
| spec_p0.5 | 8 | (+0.500) 3.5e-09 | (-0.023,-0.155) 3.1e-01 | (+0.732,-0.743,-0.084,+0.112) 2.3e-01 | (+1.799,-0.210,+0.018) 9.7e-02 |
| spec_p0.5 | 16 | (+0.500) 2.1e-09 | (-0.023,-0.160) 2.9e-01 | (+0.708,-0.724,+0.023,+0.220) 3.6e-01 | (+1.837,-0.258,+0.026) 9.4e-02 |
| spec_p0.5 | 64 | (+0.500) 4.5e-09 | (-0.041,-0.160) 3.1e-01 | (+0.158,-0.143,+0.154,+0.320) 3.9e-01 | (+1.888,-0.638,+0.148) 1.3e-01 |
| dpre_-.25_-.25 | 1 | (+0.430) 5.2e-01 | (-0.250,-0.238) 2.2e-16 | (-0.125,-0.125,-0.461,-0.079) 1.9e-16 | (+2.949,-7.198,+3.635) 5.2e-01 |
| dpre_-.25_-.25 | 2 | (+0.630) 5.6e-01 | (-0.250,-0.250) 1.8e-16 | (+0.499,-0.746,-0.610,-0.061) 6.3e-02 | (+1.135,-1.268,+1.128) 5.6e-01 |
| dpre_-.25_-.25 | 4 | (+0.617) 5.3e-01 | (-0.250,-0.250) 2.0e-16 | (+0.615,-0.849,-0.415,-0.029) 2.7e-01 | (+0.573,-1.512,+1.467) 5.3e-01 |
| dpre_-.25_-.25 | 8 | (+0.610) 5.2e-01 | (-0.250,-0.250) 2.7e-16 | (+0.626,-0.850,-0.340,+0.064) 4.0e-01 | (+1.404,-0.182,+0.026) 5.2e-01 |
| dpre_-.25_-.25 | 16 | (+0.687) 4.9e-01 | (-0.250,-0.250) 4.6e-16 | (+0.574,-0.803,-0.129,+0.234) 5.1e-01 | (+1.296,-0.265,+0.084) 4.9e-01 |
| dpre_-.25_-.25 | 64 | (+0.614) 5.0e-01 | (-0.250,-0.250) 3.4e-16 | (+0.262,-0.508,+0.011,+0.289) 6.6e-01 | (+1.061,-1.702,+0.715) 4.8e-01 |
| cpolar_rowwhite | 1 | (-0.300) 8.6e-01 | (-0.500,-0.705) 5.5e-16 | (-0.256,-0.244,+0.084,-0.169) 5.7e-16 | (+1.500,-0.500,+0.000) 8.6e-01 |
| cpolar_rowwhite | 2 | (+0.112) 9.1e-01 | (-0.561,-0.284) 4.5e-01 | (-0.000,-0.500,-0.000,-0.000) 8.5e-16 | (+2.985,-0.837,+0.089) 9.1e-01 |
| cpolar_rowwhite | 4 | (+0.044) 8.7e-01 | (-0.560,-0.312) 5.2e-01 | (+0.000,-0.500,-0.000,+0.000) 3.4e-15 | (+2.610,-2.984,+1.406) 8.7e-01 |
| cpolar_rowwhite | 8 | (+0.003) 8.2e-01 | (-0.536,-0.326) 5.7e-01 | (-0.000,-0.500,+0.000,+0.000) 8.6e-15 | (+3.029,-6.405,+4.768) 8.2e-01 |
| cpolar_rowwhite | 16 | (+0.000) 8.0e-01 | (-0.525,-0.334) 6.2e-01 | (-0.000,-0.500,-0.000,+0.000) 2.0e-14 | (+2.738,-8.281,+9.672) 8.0e-01 |
| cpolar_rowwhite | 64 | (-0.025) 7.5e-01 | (-0.506,-0.362) 6.6e-01 | (-0.000,-0.500,+0.000,+0.000) 1.8e-13 | (+2.735,-15.829,+35.430) 7.9e-01 |
| cpolar_colwhite | 1 | (-0.287) 2.7e-16 | (-0.000,-0.161) 2.2e-16 | (-0.459,+0.459,-0.106,-0.626) 2.3e-16 | (+3.558,-5.424,+0.972) 2.6e-16 |
| cpolar_colwhite | 2 | (+0.000) 4.4e-01 | (-0.024,-0.501) 6.2e-01 | (-0.000,+0.000,-0.500,-0.000) 9.7e-14 | (+2.919,-4.957,+2.991) 4.4e-01 |
| cpolar_colwhite | 4 | (+0.000) 3.2e-01 | (-0.043,-0.551) 6.6e-01 | (-0.000,-0.000,-0.500,+0.000) 1.2e-13 | (+2.778,-3.796,+2.002) 3.2e-01 |
| cpolar_colwhite | 8 | (-0.003) 3.0e-01 | (-0.057,-0.518) 6.7e-01 | (-0.000,+0.000,-0.500,+0.000) 3.3e-14 | (+2.969,-6.185,+4.499) 3.0e-01 |
| cpolar_colwhite | 16 | (-0.000) 2.8e-01 | (-0.052,-0.517) 6.7e-01 | (-0.000,+0.000,-0.500,+0.000) 2.1e-14 | (+2.986,-8.080,+7.677) 5.3e-01 |
| cpolar_colwhite | 64 | (-0.001) 2.5e-01 | (-0.102,-0.508) 6.7e-01 | (+0.000,-0.000,-0.500,-0.000) 1.3e-13 | (+2.949,-19.228,+44.329) 7.2e-01 |
| cpolar_dshampoo | 1 | (-0.300) 5.2e-01 | (-0.250,+0.054) 1.5e-16 | (-0.125,-0.125,-0.461,-0.079) 2.1e-16 | (+2.046,-5.414,+3.857) 5.2e-01 |
| cpolar_dshampoo | 2 | (+0.038) 5.8e-01 | (-0.317,-0.360) 6.4e-01 | (-0.000,-0.250,-0.250,+0.000) 4.8e-15 | (+2.960,-4.675,+2.716) 5.8e-01 |
| cpolar_dshampoo | 4 | (+0.008) 4.8e-01 | (-0.301,-0.404) 5.8e-01 | (-0.000,-0.250,-0.250,+0.000) 8.5e-15 | (+2.746,-4.784,+3.055) 4.8e-01 |
| cpolar_dshampoo | 8 | (+0.002) 4.8e-01 | (-0.284,-0.414) 6.5e-01 | (-0.000,-0.250,-0.250,+0.000) 4.1e-14 | (+2.887,-5.935,+4.379) 4.8e-01 |
| cpolar_dshampoo | 16 | (-0.003) 4.7e-01 | (-0.276,-0.432) 6.8e-01 | (-0.000,-0.250,-0.250,+0.000) 3.8e-14 | (+2.611,-6.721,+6.955) 4.7e-01 |
| cpolar_dshampoo | 64 | (-0.003) 4.1e-01 | (-0.273,-0.416) 6.6e-01 | (-0.000,-0.250,-0.250,-0.000) 3.4e-14 | (+2.825,-14.979,+30.037) 4.1e-01 |
| cpolar(.3,-.4,-.2,.1) | 1 | (-0.345) 2.1e-01 | (-0.100,+0.421) 3.0e-16 | (+0.431,-0.531,-0.626,-0.101) 2.1e-16 | (+1.354,-3.123,+2.740) 2.1e-01 |
| cpolar(.3,-.4,-.2,.1) | 2 | (+0.363) 2.2e-01 | (-0.106,-0.125) 2.7e-01 | (+0.300,-0.400,-0.200,+0.100) 2.1e-15 | (+0.876,-3.906,+3.976) 2.2e-01 |
| cpolar(.3,-.4,-.2,.1) | 4 | (+0.267) 2.6e-01 | (-0.120,-0.191) 3.6e-01 | (+0.300,-0.400,-0.200,+0.100) 1.1e-14 | (+2.836,-0.593,+0.050) 2.5e-01 |
| cpolar(.3,-.4,-.2,.1) | 8 | (+0.179) 2.6e-01 | (-0.119,-0.231) 5.1e-01 | (+0.300,-0.400,-0.200,+0.100) 1.2e-14 | (+2.512,-3.067,+1.606) 2.6e-01 |
| cpolar(.3,-.4,-.2,.1) | 16 | (+0.132) 2.8e-01 | (-0.115,-0.277) 6.1e-01 | (+0.300,-0.400,-0.200,+0.100) 2.6e-14 | (+2.498,-4.070,+2.845) 2.8e-01 |
| cpolar(.3,-.4,-.2,.1) | 64 | (+0.115) 2.9e-01 | (-0.108,-0.282) 5.9e-01 | (+0.300,-0.400,-0.200,+0.100) 2.0e-14 | (+2.603,-8.745,+12.210) 3.0e-01 |

## generator = sketch


### eps = 1e-06: fixed candidates confusable with the true rule

| true rule | r=1 | r=2 | r=4 | r=8 | r=16 | r=64 |
|---|---|---|---|---|---|---|
| frob | cpolar_colwhite, col_norm, polar, ns_cubic5, ns_cubic15, ns_quintic8, ns_quintic5, apolar_-.5, spec_p0.5, ns_quintic3, ns_quintic5_f32 | - | - | - | - | - |
| sign | cpolar_rowwhite, row_norm | - | - | - | - | - |
| row_norm | cpolar_rowwhite, sign | - | - | - | - | - |
| col_norm | frob, apolar_-.5, cpolar_colwhite, ns_quintic5, spec_p0.5, polar, ns_cubic5, ns_cubic15, ns_quintic3, ns_quintic8, ns_quintic5_f32 | - | - | - | - | - |
| polar | spec_p0.5, ns_quintic3, ns_quintic5, frob, col_norm, ns_cubic5, ns_cubic15, cpolar_colwhite, ns_quintic8, apolar_-.5, ns_quintic5_f32 | ns_cubic15 | ns_cubic15 | - | - | - |
| ns_quintic5 | polar, ns_cubic5, ns_cubic15, spec_p0.5, frob, col_norm, cpolar_colwhite, ns_quintic8, apolar_-.5, ns_quintic3, ns_quintic5_f32 | ns_quintic5_f32 | - | ns_quintic5_f32 | - | - |
| ns_quintic3 | ns_quintic5, frob, col_norm, apolar_-.5, cpolar_colwhite, ns_quintic8, spec_p0.5, polar, ns_cubic5, ns_cubic15, ns_quintic5_f32 | - | - | - | - | - |
| ns_quintic8 | polar, ns_cubic5, ns_cubic15, frob, col_norm, spec_p0.5, ns_quintic3, apolar_-.5, ns_quintic5, cpolar_colwhite, ns_quintic5_f32 | - | - | - | - | - |
| ns_cubic5 | polar, ns_cubic15, ns_quintic8, ns_quintic5, ns_quintic3, frob, col_norm, apolar_-.5, spec_p0.5, cpolar_colwhite, ns_quintic5_f32 | - | - | - | - | - |
| ns_cubic15 | polar, ns_cubic5, ns_quintic5, spec_p0.5, cpolar_colwhite, frob, col_norm, ns_quintic8, apolar_-.5, ns_quintic3, ns_quintic5_f32 | polar | polar | polar | - | - |
| spec_p0.5 | polar, ns_cubic5, ns_cubic15, ns_quintic8, ns_quintic5, frob, col_norm, apolar_-.5, cpolar_colwhite, ns_quintic3, ns_quintic5_f32 | - | - | - | - | - |
| dpre_-.25_-.25 | cpolar_dshampoo | - | - | - | - | - |
| cpolar_rowwhite | sign, row_norm | - | - | - | - | - |
| cpolar_colwhite | apolar_-.5, polar, ns_cubic5, ns_cubic15, ns_quintic8, ns_quintic3, frob, col_norm, ns_quintic5, spec_p0.5, ns_quintic5_f32 | - | - | - | - | - |
| cpolar_dshampoo | dpre_-.25_-.25 | - | - | - | - | - |
| ns_quintic5_f32 | cpolar_colwhite, ns_quintic3, frob, col_norm, polar, ns_cubic5, ns_cubic15, ns_quintic5, apolar_-.5, ns_quintic8, spec_p0.5 | ns_quintic5 | ns_quintic5 | ns_quintic5 | - | - |
| ns_quintic5_bf16 | - | - | - | - | - | - |
| apolar_-.5 | frob, col_norm, cpolar_colwhite, ns_quintic3, ns_quintic8, polar, ns_cubic5, ns_cubic15, ns_quintic5, spec_p0.5, ns_quintic5_f32 | - | - | - | - | - |
| cpolar(.3,-.4,-.2,.1) | - | - | - | - | - | - |

### eps = 0.001: fixed candidates confusable with the true rule

| true rule | r=1 | r=2 | r=4 | r=8 | r=16 | r=64 |
|---|---|---|---|---|---|---|
| frob | cpolar_colwhite, col_norm, polar, ns_cubic5, ns_cubic15, ns_quintic8, ns_quintic5, apolar_-.5, spec_p0.5, ns_quintic3, ns_quintic5_f32 | - | - | - | - | - |
| sign | cpolar_rowwhite, row_norm | - | - | - | - | - |
| row_norm | cpolar_rowwhite, sign | - | - | - | - | - |
| col_norm | frob, apolar_-.5, cpolar_colwhite, ns_quintic5, spec_p0.5, polar, ns_cubic5, ns_cubic15, ns_quintic3, ns_quintic8, ns_quintic5_f32 | - | - | - | - | - |
| polar | spec_p0.5, ns_quintic3, ns_quintic5, frob, col_norm, ns_cubic5, ns_cubic15, cpolar_colwhite, ns_quintic8, apolar_-.5, ns_quintic5_f32 | ns_cubic15, ns_cubic5 | ns_cubic15 | ns_cubic15 | - | - |
| ns_quintic5 | polar, ns_cubic5, ns_cubic15, spec_p0.5, frob, col_norm, cpolar_colwhite, ns_quintic8, apolar_-.5, ns_quintic3, ns_quintic5_f32 | ns_quintic5_f32 | ns_quintic5_f32 | ns_quintic5_f32 | ns_quintic5_f32 | ns_quintic5_f32 |
| ns_quintic3 | ns_quintic5, frob, col_norm, apolar_-.5, cpolar_colwhite, ns_quintic8, spec_p0.5, polar, ns_cubic5, ns_cubic15, ns_quintic5_f32 | - | - | - | - | - |
| ns_quintic8 | polar, ns_cubic5, ns_cubic15, frob, col_norm, spec_p0.5, ns_quintic3, apolar_-.5, ns_quintic5, cpolar_colwhite, ns_quintic5_f32 | - | - | - | - | - |
| ns_cubic5 | polar, ns_cubic15, ns_quintic8, ns_quintic5, ns_quintic3, frob, col_norm, apolar_-.5, spec_p0.5, cpolar_colwhite, ns_quintic5_f32 | polar, ns_cubic15 | - | - | - | - |
| ns_cubic15 | polar, ns_cubic5, ns_quintic5, spec_p0.5, cpolar_colwhite, frob, col_norm, ns_quintic8, apolar_-.5, ns_quintic3, ns_quintic5_f32 | polar | polar | polar | - | - |
| spec_p0.5 | polar, ns_cubic5, ns_cubic15, ns_quintic8, ns_quintic5, frob, col_norm, apolar_-.5, cpolar_colwhite, ns_quintic3, ns_quintic5_f32 | - | - | - | - | - |
| dpre_-.25_-.25 | cpolar_dshampoo | - | - | - | - | - |
| cpolar_rowwhite | sign, row_norm | - | - | - | - | - |
| cpolar_colwhite | apolar_-.5, polar, ns_cubic5, ns_cubic15, ns_quintic8, ns_quintic3, frob, col_norm, ns_quintic5, spec_p0.5, ns_quintic5_f32 | - | - | - | - | - |
| cpolar_dshampoo | dpre_-.25_-.25 | - | - | - | - | - |
| ns_quintic5_f32 | cpolar_colwhite, ns_quintic3, frob, col_norm, polar, ns_cubic5, ns_cubic15, ns_quintic5, apolar_-.5, ns_quintic8, spec_p0.5 | ns_quintic5 | ns_quintic5 | ns_quintic5 | ns_quintic5 | ns_quintic5 |
| ns_quintic5_bf16 | - | - | - | - | - | - |
| apolar_-.5 | frob, col_norm, cpolar_colwhite, ns_quintic3, ns_quintic8, polar, ns_cubic5, ns_cubic15, ns_quintic5, spec_p0.5, ns_quintic5_f32 | polar, ns_cubic15, ns_cubic5 | - | - | - | - |
| cpolar(.3,-.4,-.2,.1) | - | - | - | - | - | - |

### eps = 0.01: fixed candidates confusable with the true rule

| true rule | r=1 | r=2 | r=4 | r=8 | r=16 | r=64 |
|---|---|---|---|---|---|---|
| frob | cpolar_colwhite, col_norm, polar, ns_cubic5, ns_cubic15, ns_quintic8, ns_quintic5, apolar_-.5, spec_p0.5, ns_quintic3, ns_quintic5_f32 | - | - | - | - | - |
| sign | cpolar_rowwhite, row_norm | - | - | - | - | - |
| row_norm | cpolar_rowwhite, sign | - | - | - | - | - |
| col_norm | frob, apolar_-.5, cpolar_colwhite, ns_quintic5, spec_p0.5, polar, ns_cubic5, ns_cubic15, ns_quintic3, ns_quintic8, ns_quintic5_f32 | - | - | - | - | - |
| polar | spec_p0.5, ns_quintic3, ns_quintic5, frob, col_norm, ns_cubic5, ns_cubic15, cpolar_colwhite, ns_quintic8, apolar_-.5, ns_quintic5_f32 | ns_cubic15, ns_cubic5, apolar_-.5 | ns_cubic15, apolar_-.5 | ns_cubic15 | - | - |
| ns_quintic5 | polar, ns_cubic5, ns_cubic15, spec_p0.5, frob, col_norm, cpolar_colwhite, ns_quintic8, apolar_-.5, ns_quintic3, ns_quintic5_f32 | ns_quintic5_f32 | ns_quintic5_f32 | ns_quintic5_f32 | ns_quintic5_f32 | ns_quintic5_f32 |
| ns_quintic3 | ns_quintic5, frob, col_norm, apolar_-.5, cpolar_colwhite, ns_quintic8, spec_p0.5, polar, ns_cubic5, ns_cubic15, ns_quintic5_f32 | - | - | - | - | - |
| ns_quintic8 | polar, ns_cubic5, ns_cubic15, frob, col_norm, spec_p0.5, ns_quintic3, apolar_-.5, ns_quintic5, cpolar_colwhite, ns_quintic5_f32 | - | - | - | - | - |
| ns_cubic5 | polar, ns_cubic15, ns_quintic8, ns_quintic5, ns_quintic3, frob, col_norm, apolar_-.5, spec_p0.5, cpolar_colwhite, ns_quintic5_f32 | polar, ns_cubic15, apolar_-.5 | - | - | - | - |
| ns_cubic15 | polar, ns_cubic5, ns_quintic5, spec_p0.5, cpolar_colwhite, frob, col_norm, ns_quintic8, apolar_-.5, ns_quintic3, ns_quintic5_f32 | polar, apolar_-.5 | polar, apolar_-.5 | polar | - | - |
| spec_p0.5 | polar, ns_cubic5, ns_cubic15, ns_quintic8, ns_quintic5, frob, col_norm, apolar_-.5, cpolar_colwhite, ns_quintic3, ns_quintic5_f32 | - | - | - | - | - |
| dpre_-.25_-.25 | cpolar_dshampoo | - | - | - | - | - |
| cpolar_rowwhite | sign, row_norm | - | - | - | - | - |
| cpolar_colwhite | apolar_-.5, polar, ns_cubic5, ns_cubic15, ns_quintic8, ns_quintic3, frob, col_norm, ns_quintic5, spec_p0.5, ns_quintic5_f32 | - | - | - | - | - |
| cpolar_dshampoo | dpre_-.25_-.25 | - | - | - | - | - |
| ns_quintic5_f32 | cpolar_colwhite, ns_quintic3, frob, col_norm, polar, ns_cubic5, ns_cubic15, ns_quintic5, apolar_-.5, ns_quintic8, spec_p0.5 | ns_quintic5 | ns_quintic5 | ns_quintic5 | ns_quintic5 | ns_quintic5 |
| ns_quintic5_bf16 | - | - | - | - | - | - |
| apolar_-.5 | frob, col_norm, cpolar_colwhite, ns_quintic3, ns_quintic8, polar, ns_cubic5, ns_cubic15, ns_quintic5, spec_p0.5, ns_quintic5_f32 | polar, ns_cubic15, ns_cubic5 | polar, ns_cubic15 | - | - | - |
| cpolar(.3,-.4,-.2,.1) | - | - | - | - | - | - |

### noise-free misfit to the NEAREST other fixed candidate (log10)

| true rule | r=1 | r=2 | r=4 | r=8 | r=16 | r=64 |
|---|---|---|---|---|---|---|
| frob | -16.1 (cpolar_colwhite) | -0.8 (spec_p0.5) | -0.7 (spec_p0.5) | -0.5 (spec_p0.5) | -0.5 (spec_p0.5) | -0.5 (spec_p0.5) |
| sign | -15.4 (cpolar_rowwhite) | -0.4 (row_norm) | -0.3 (row_norm) | -0.2 (row_norm) | -0.2 (row_norm) | -0.2 (row_norm) |
| row_norm | -15.4 (cpolar_rowwhite) | -0.7 (cpolar_rowwhite) | -0.4 (cpolar_rowwhite) | -0.4 (dpre_-.25_-.25) | -0.4 (dpre_-.25_-.25) | -0.4 (dpre_-.25_-.25) |
| col_norm | -15.9 (frob) | -0.7 (ns_quintic8) | -0.5 (spec_p0.5) | -0.5 (spec_p0.5) | -0.4 (frob) | -0.5 (frob) |
| polar | -16.0 (spec_p0.5) | -15.7 (ns_cubic15) | -15.8 (ns_cubic15) | -3.4 (ns_cubic15) | -1.6 (apolar_-.5) | -1.3 (apolar_-.5) |
| ns_quintic5 | -15.9 (ns_cubic15) | -6.2 (ns_quintic5_f32) | -5.9 (ns_quintic5_f32) | -6.1 (ns_quintic5_f32) | -5.6 (ns_quintic5_f32) | -5.3 (ns_quintic5_f32) |
| ns_quintic3 | -15.9 (ns_quintic5) | -1.1 (ns_quintic5_bf16) | -1.2 (ns_quintic5_bf16) | -0.8 (ns_quintic5_bf16) | -0.4 (spec_p0.5) | -0.4 (spec_p0.5) |
| ns_quintic8 | -15.8 (ns_cubic15) | -0.8 (spec_p0.5) | -0.8 (ns_cubic15) | -0.7 (ns_cubic15) | -0.7 (polar) | -0.8 (polar) |
| ns_cubic5 | -15.9 (ns_cubic15) | -4.5 (ns_cubic15) | -0.7 (ns_cubic15) | -0.6 (spec_p0.5) | -0.6 (spec_p0.5) | -0.6 (spec_p0.5) |
| ns_cubic15 | -16.4 (ns_cubic5) | -16.0 (polar) | -16.2 (polar) | -9.5 (polar) | -1.3 (polar) | -0.8 (ns_quintic5_f32) |
| spec_p0.5 | -16.1 (ns_cubic15) | -0.8 (frob) | -0.7 (frob) | -0.7 (ns_cubic5) | -0.6 (ns_cubic5) | -0.6 (ns_cubic5) |
| dpre_-.25_-.25 | -15.8 (cpolar_dshampoo) | -0.6 (cpolar_dshampoo) | -0.4 (cpolar_dshampoo) | -0.3 (row_norm) | -0.3 (col_norm) | -0.4 (col_norm) |
| cpolar_rowwhite | -15.6 (row_norm) | -0.6 (row_norm) | -0.4 (row_norm) | -0.4 (cpolar_dshampoo) | -0.3 (cpolar_dshampoo) | -0.4 (cpolar_dshampoo) |
| cpolar_colwhite | -15.8 (apolar_-.5) | -1.2 (apolar_-.5) | -0.9 (ns_cubic15) | -0.8 (ns_cubic15) | -0.6 (polar) | -0.6 (polar) |
| cpolar_dshampoo | -15.6 (dpre_-.25_-.25) | -0.5 (dpre_-.25_-.25) | -0.4 (dpre_-.25_-.25) | -0.4 (polar) | -0.4 (polar) | -0.4 (cpolar_colwhite) |
| ns_quintic5_f32 | -6.5 (cpolar_colwhite) | -6.2 (ns_quintic5) | -6.0 (ns_quintic5) | -6.0 (ns_quintic5) | -5.5 (ns_quintic5) | -5.2 (ns_quintic5) |
| ns_quintic5_bf16 | -1.7 (apolar_-.5) | -1.6 (ns_quintic5_f32) | -1.5 (ns_quintic5) | -1.3 (ns_quintic5_f32) | -0.8 (ns_quintic5) | -0.7 (ns_quintic5_f32) |
| apolar_-.5 | -16.0 (col_norm) | -3.3 (ns_cubic15) | -2.1 (ns_cubic15) | -1.8 (ns_cubic15) | -1.6 (polar) | -1.3 (polar) |
| cpolar(.3,-.4,-.2,.1) | -0.8 (ns_quintic5_f32) | -0.7 (spec_p0.5) | -0.7 (spec_p0.5) | -0.6 (ns_cubic15) | -0.6 (ns_cubic15) | -0.5 (polar) |

### noise-free family fits (fitted params; held-out misfit)

| true rule | r | FIT spec_power | FIT diag_precond | FIT cpolar | FIT ns5_coeffs | FIT apolar |
|---|---|---|---|---|---|---|
| frob | 1 | (+1.182) 2.1e-16 | (-0.000,-1.012) 1.3e-16 | (-0.458,+0.458,-0.620,-0.622) 2.0e-16 | (+1.551,-1.609,+1.141) 3.4e-16 | (+0.391,-0.610) 2.0e-16 |
| frob | 2 | (+1.000) 3.5e-09 | (-0.000,+0.000) 1.8e-16 | (+0.745,-0.744,-0.435,+0.103) 6.3e-02 | (+0.647,-0.000,+0.000) 2.6e-09 | (-2.702,-2.702) 3.4e-01 |
| frob | 4 | (+1.000) 8.8e-10 | (-0.000,+0.000) 2.7e-16 | (+0.905,-0.899,-0.315,+0.195) 1.8e-01 | (+0.574,-0.000,+0.000) 1.5e-09 | (+0.292,+0.292) 5.0e-01 |
| frob | 8 | (+1.000) 1.6e-09 | (+0.000,+0.000) 3.2e-16 | (+0.944,-0.916,-0.029,+0.293) 3.6e-01 | (+0.579,-0.000,+0.000) 4.1e-09 | (+3.166,+3.166) 6.8e-01 |
| frob | 16 | (+1.000) 6.5e-09 | (+0.000,+0.000) 4.4e-16 | (+1.001,-0.955,+0.069,+0.467) 5.8e-01 | (+0.448,-0.000,+0.000) 9.4e-09 | (+2.372,+2.372) 8.3e-01 |
| frob | 64 | (+1.000) 7.2e-09 | (+0.000,+0.000) 4.8e-16 | (+0.918,-0.835,+0.944,+1.132) 8.7e-01 | (+0.816,+0.000,-0.000) 2.3e-09 | (+2.980,+2.987) 9.5e-01 |
| row_norm | 1 | (-0.300) 8.2e-01 | (-0.500,-0.104) 1.5e-16 | (-0.232,-0.268,+0.357,+0.119) 8.1e-16 | (+2.949,-7.193,+3.635) 8.2e-01 | (-0.600,-0.600) 8.2e-01 |
| row_norm | 2 | (+0.530) 7.8e-01 | (-0.500,-0.000) 1.2e-16 | (+0.310,-0.807,-0.011,+0.145) 9.4e-02 | (+2.426,-1.331,+0.324) 7.8e-01 | (-1.028,-2.550) 7.9e-01 |
| row_norm | 4 | (+0.635) 7.4e-01 | (-0.500,-0.000) 1.7e-16 | (+0.418,-0.902,-0.160,+0.189) 1.7e-01 | (+0.000,-0.000,+0.000) 7.5e-01 | (-0.147,-0.819) 8.0e-01 |
| row_norm | 8 | (+0.705) 7.3e-01 | (-0.500,-0.000) 1.8e-16 | (+0.496,-0.959,+0.322,+0.394) 4.1e-01 | (+0.928,-0.362,+0.085) 7.3e-01 | (-1.088,-0.809) 8.3e-01 |
| row_norm | 16 | (+0.757) 7.1e-01 | (-0.500,+0.000) 1.5e-16 | (+0.509,-0.936,+0.296,+0.497) 6.7e-01 | (+0.850,-0.316,+0.032) 7.1e-01 | (+2.093,+2.222) 9.0e-01 |
| row_norm | 64 | (+0.788) 7.2e-01 | (-0.500,+0.000) 1.7e-16 | (+0.400,-0.819,+1.173,+1.233) 9.0e-01 | (+1.233,-0.155,+0.041) 7.2e-01 | (+3.352,+3.359) 9.7e-01 |
| col_norm | 1 | (-0.300) 1.9e-16 | (+0.000,-0.151) 1.3e-16 | (-0.458,+0.458,-0.622,-0.621) 1.7e-16 | (+1.859,-6.942,+3.815) 1.9e-16 | (-0.617,-0.098) 1.5e-16 |
| col_norm | 2 | (+0.511) 2.2e-01 | (+0.000,-0.500) 1.7e-16 | (+0.750,-0.746,-1.284,-0.345) 6.5e-02 | (+0.822,-2.597,+2.904) 4.1e-01 | (+16.664,+7.355) 3.5e-01 |
| col_norm | 4 | (+0.741) 2.8e-01 | (-0.000,-0.500) 1.3e-16 | (+0.976,-0.968,-1.131,-0.249) 1.7e-01 | (+1.406,-0.068,+0.007) 2.8e-01 | (+1.884,+1.334) 4.7e-01 |
| col_norm | 8 | (+0.767) 3.3e-01 | (-0.000,-0.500) 2.2e-16 | (+0.906,-0.885,-0.641,-0.168) 3.8e-01 | (+0.971,-0.574,+0.803) 3.3e-01 | (-1.258,-1.040) 6.6e-01 |
| col_norm | 16 | (+0.774) 3.6e-01 | (-0.000,-0.500) 1.8e-16 | (+1.000,-0.939,-0.623,-0.119) 6.3e-01 | (+0.442,-0.509,+0.478) 3.5e-01 | (+2.170,+2.095) 8.2e-01 |
| col_norm | 64 | (+0.789) 3.3e-01 | (-0.000,-0.500) 2.1e-16 | (+0.823,-0.744,-0.647,-0.171) 9.0e-01 | (+1.047,-0.196,+0.035) 3.2e-01 | (+2.446,+2.684) 9.5e-01 |
| polar | 1 | (-0.307) 1.7e-16 | (-0.000,+0.642) 2.5e-16 | (+0.458,-0.458,+0.415,+0.428) 3.0e-16 | (+1.575,-1.660,+1.149) 1.4e-16 | (-0.106,-0.093) 1.6e-16 |
| polar | 2 | (-0.000) 1.1e-15 | (-0.019,-0.468) 2.1e-01 | (-0.000,+0.000,-0.000,+0.000) 3.7e-16 | (+1.630,-2.796,+1.759) 3.1e-12 | (+0.000,-0.000) 3.9e-16 |
| polar | 4 | (-0.000) 2.6e-09 | (-0.040,-0.461) 4.7e-01 | (+0.000,-0.000,+0.000,-0.000) 2.5e-15 | (+2.284,-3.898,+2.783) 4.3e-01 | (+0.000,-0.000) 4.3e-15 |
| polar | 8 | (+0.000) 7.9e-09 | (-0.062,-0.394) 6.5e-01 | (+0.000,-0.000,-0.000,-0.000) 7.7e-14 | (+2.572,-4.557,+3.279) 5.0e-02 | (+0.000,+0.000) 6.1e-14 |
| polar | 16 | (-0.000) 2.7e-09 | (-0.058,-0.342) 8.1e-01 | (-0.000,-0.000,-0.000,+0.000) 7.0e-13 | (+3.032,-7.113,+6.009) 8.8e-02 | (-0.000,+0.000) 9.4e-13 |
| polar | 64 | (-0.000) 3.7e-09 | (-0.066,-0.313) 9.5e-01 | (-0.000,+0.000,-0.000,-0.000) 4.5e-07 | (+3.887,-7.546,+4.106) 4.0e-01 | (-0.000,+0.000) 6.2e-08 |
| ns_quintic5 | 1 | (-0.306) 3.2e-16 | (-0.000,-0.930) 1.5e-16 | (-0.458,+0.458,-0.622,+0.414) 2.0e-16 | (+1.725,-5.907,+3.972) 2.1e-16 | (+0.394,-0.103) 2.3e-16 |
| ns_quintic5 | 2 | (-0.417) 1.8e-01 | (-0.021,-0.669) 3.4e-01 | (-0.052,+0.051,+0.050,-0.172) 1.8e-01 | (+3.444,-4.775,+2.032) 7.4e-09 | (-1.058,+2.904) 1.8e-01 |
| ns_quintic5 | 4 | (+0.014) 1.5e-01 | (-0.053,-0.457) 4.7e-01 | (+0.055,-0.054,-0.005,-0.017) 1.6e-01 | (+3.445,-4.775,+2.032) 6.3e-09 | (-0.047,-0.420) 1.5e-01 |
| ns_quintic5 | 8 | (+0.027) 1.6e-01 | (-0.055,-0.378) 6.1e-01 | (+0.057,-0.056,-0.006,+0.003) 1.6e-01 | (+3.444,-4.775,+2.032) 2.9e-09 | (+0.003,+0.177) 1.6e-01 |
| ns_quintic5 | 16 | (-0.026) 1.8e-01 | (-0.048,-0.337) 8.2e-01 | (-0.005,+0.005,-0.001,+0.016) 1.9e-01 | (+3.445,-4.775,+2.031) 1.0e-08 | (+0.001,-0.017) 1.9e-01 |
| ns_quintic5 | 64 | (+0.105) 3.4e-01 | (-0.060,-0.264) 9.2e-01 | (+0.114,-0.100,+0.005,+0.020) 3.8e-01 | (+3.444,-4.775,+2.032) 3.7e-09 | (+0.134,+0.381) 3.9e-01 |
| spec_p0.5 | 1 | (+0.400) 1.9e-16 | (-0.000,+0.524) 2.1e-16 | (-0.458,+0.458,+0.414,-0.622) 2.7e-16 | (+2.359,-5.929,+1.422) 9.8e-17 | (-0.105,-0.100) 1.7e-16 |
| spec_p0.5 | 2 | (+0.500) 3.5e-09 | (-0.009,-0.235) 1.4e-01 | (+0.478,-0.480,+0.125,-0.002) 4.4e-02 | (+1.449,-0.460,+0.155) 9.8e-04 | (+0.097,+0.189) 2.0e-01 |
| spec_p0.5 | 4 | (+0.500) 5.2e-10 | (-0.018,-0.198) 1.6e-01 | (+0.707,-0.710,-0.005,-0.007) 5.2e-02 | (+1.657,-0.187,+0.019) 3.0e-02 | (-0.166,-0.296) 2.4e-01 |
| spec_p0.5 | 8 | (+0.500) 3.4e-09 | (-0.024,-0.185) 2.4e-01 | (+0.826,-0.820,-0.039,+0.063) 2.0e-01 | (+1.727,-0.251,+0.030) 6.9e-02 | (+0.368,+0.594) 4.6e-01 |
| spec_p0.5 | 16 | (+0.500) 6.6e-09 | (-0.027,-0.153) 2.9e-01 | (+0.814,-0.792,+0.064,+0.232) 4.6e-01 | (+1.909,-0.249,+0.022) 1.1e-01 | (+0.279,+0.383) 6.7e-01 |
| spec_p0.5 | 64 | (+0.500) 1.8e-10 | (-0.026,-0.127) 3.1e-01 | (+0.741,-0.714,+0.430,+0.572) 7.9e-01 | (+2.565,-0.173,+0.005) 1.5e-01 | (+0.845,+1.027) 8.5e-01 |
| dpre_-.25_-.25 | 1 | (+1.157) 4.4e-01 | (-0.250,-0.003) 2.2e-16 | (-0.125,-0.125,-0.461,-0.079) 1.9e-16 | (+4.331,-2.635,+0.591) 4.4e-01 | (-0.600,-0.600) 4.4e-01 |
| dpre_-.25_-.25 | 2 | (+0.567) 4.7e-01 | (-0.250,-0.250) 2.0e-16 | (+0.535,-0.783,-0.655,-0.142) 9.3e-02 | (+2.380,-1.022,+0.200) 4.7e-01 | (+2.794,+1.969) 5.2e-01 |
| dpre_-.25_-.25 | 4 | (+0.692) 4.4e-01 | (-0.250,-0.250) 1.6e-16 | (+0.693,-0.928,-0.677,-0.077) 1.6e-01 | (+0.000,-0.000,+0.000) 4.4e-01 | (-0.047,+0.172) 5.5e-01 |
| dpre_-.25_-.25 | 8 | (+0.743) 4.6e-01 | (-0.250,-0.250) 1.7e-16 | (+0.709,-0.916,-0.279,+0.086) 4.2e-01 | (+1.261,-0.135,+0.030) 4.6e-01 | (+0.208,+0.858) 7.2e-01 |
| dpre_-.25_-.25 | 16 | (+0.758) 4.5e-01 | (-0.250,-0.250) 3.0e-16 | (+0.737,-0.918,-0.166,+0.222) 6.5e-01 | (+0.338,-0.380,+0.296) 4.5e-01 | (-0.050,-0.029) 8.3e-01 |
| dpre_-.25_-.25 | 64 | (+0.782) 4.3e-01 | (-0.250,-0.250) 3.3e-16 | (+0.632,-0.778,+0.416,+0.681) 9.1e-01 | (+1.253,-0.147,+0.034) 4.3e-01 | (+3.568,+3.565) 9.5e-01 |
| cpolar_rowwhite | 1 | (-0.300) 8.1e-01 | (-0.500,+0.417) 2.2e-16 | (-0.754,+0.254,-0.644,+0.449) 3.1e-16 | (+2.000,-1.500,+0.500) 8.1e-01 | (-0.100,-0.100) 8.1e-01 |
| cpolar_rowwhite | 2 | (+0.027) 7.8e-01 | (-0.509,-0.219) 1.5e-01 | (-0.000,-0.500,-0.000,-0.000) 4.5e-16 | (+1.215,-3.333,+3.376) 8.7e-01 | (+0.768,-0.322) 7.8e-01 |
| cpolar_rowwhite | 4 | (+0.013) 7.4e-01 | (-0.532,-0.314) 3.7e-01 | (+0.000,-0.500,-0.000,+0.000) 2.1e-15 | (+2.444,-0.603,+0.063) 7.4e-01 | (+0.023,-0.018) 7.4e-01 |
| cpolar_rowwhite | 8 | (+0.008) 7.3e-01 | (-0.545,-0.329) 5.6e-01 | (+0.000,-0.500,+0.000,-0.000) 2.5e-14 | (+2.771,-5.066,+3.453) 7.3e-01 | (+0.007,+0.122) 7.3e-01 |
| cpolar_rowwhite | 16 | (+0.003) 7.3e-01 | (-0.544,-0.296) 7.8e-01 | (+0.000,-0.500,+0.000,+0.000) 6.3e-13 | (+3.098,-7.916,+7.058) 8.0e-01 | (+0.015,-0.036) 7.3e-01 |
| cpolar_rowwhite | 64 | (-0.009) 7.0e-01 | (-0.538,-0.254) 9.4e-01 | (-0.000,-0.500,+0.000,+0.000) 8.0e-12 | (+3.830,-7.180,+3.822) 7.2e-01 | (-0.020,-0.011) 7.0e-01 |
| cpolar_colwhite | 1 | (-0.300) 2.0e-16 | (-0.000,-0.928) 1.6e-16 | (-0.458,+0.458,-0.620,-0.104) 1.9e-16 | (+3.546,-6.657,+1.564) 2.6e-16 | (-0.616,-0.100) 1.3e-16 |
| cpolar_colwhite | 2 | (+0.000) 6.2e-02 | (-0.016,-0.493) 3.1e-01 | (-0.000,+0.000,-0.500,+0.000) 5.3e-16 | (+1.646,-2.784,+1.793) 6.2e-02 | (+3.298,-0.000) 6.3e-02 |
| cpolar_colwhite | 4 | (+0.002) 1.1e-01 | (-0.033,-0.533) 4.0e-01 | (-0.000,+0.000,-0.500,-0.000) 1.4e-15 | (+1.603,-4.316,+3.203) 1.1e-01 | (-1.035,+0.007) 1.2e-01 |
| cpolar_colwhite | 8 | (+0.002) 1.5e-01 | (-0.041,-0.533) 6.4e-01 | (-0.000,+0.000,-0.500,+0.000) 8.9e-14 | (+2.723,-6.538,+5.796) 1.6e-01 | (+0.399,-0.006) 1.5e-01 |
| cpolar_colwhite | 16 | (+0.000) 2.4e-01 | (-0.056,-0.601) 7.9e-01 | (+0.000,+0.000,-0.500,-0.000) 1.1e-12 | (+3.093,-7.523,+6.438) 2.6e-01 | (+0.352,-0.044) 2.4e-01 |
| cpolar_colwhite | 64 | (-0.006) 2.3e-01 | (-0.051,-0.685) 9.4e-01 | (-0.000,+0.000,-0.500,+0.000) 6.5e-10 | (+3.943,-7.780,+4.221) 3.9e-01 | (-0.005,-0.008) 2.3e-01 |
| cpolar_dshampoo | 1 | (-0.304) 4.6e-01 | (-0.250,-0.146) 1.1e-16 | (-0.125,-0.125,-0.477,-0.461) 1.8e-16 | (+1.833,-3.240,+2.849) 4.6e-01 | (+0.400,-0.100) 4.6e-01 |
| cpolar_dshampoo | 2 | (-0.002) 4.6e-01 | (-0.261,-0.375) 2.9e-01 | (+0.000,-0.250,-0.250,+0.000) 8.2e-16 | (+1.619,-5.496,+5.018) 5.5e-01 | (+0.679,-0.006) 4.6e-01 |
| cpolar_dshampoo | 4 | (+0.006) 4.2e-01 | (-0.278,-0.416) 3.9e-01 | (+0.000,-0.250,-0.250,-0.000) 1.5e-15 | (+1.838,-2.831,+0.958) 4.2e-01 | (-0.201,-0.033) 4.2e-01 |
| cpolar_dshampoo | 8 | (+0.006) 4.2e-01 | (-0.291,-0.443) 5.9e-01 | (+0.000,-0.250,-0.250,-0.000) 2.6e-14 | (+2.620,-4.953,+3.775) 5.9e-01 | (-0.096,+0.009) 4.2e-01 |
| cpolar_dshampoo | 16 | (+0.000) 4.4e-01 | (-0.294,-0.423) 7.8e-01 | (-0.000,-0.250,-0.250,+0.000) 8.3e-13 | (+3.073,-9.112,+9.622) 6.2e-01 | (-0.104,+0.007) 4.4e-01 |
| cpolar_dshampoo | 64 | (-0.004) 4.0e-01 | (-0.287,-0.463) 9.4e-01 | (-0.000,-0.250,-0.250,+0.000) 2.7e-09 | (+3.964,-7.281,+3.654) 5.2e-01 | (-0.022,-0.004) 4.0e-01 |
| apolar_-.5 | 1 | (-0.300) 1.5e-16 | (+0.000,-0.712) 1.2e-16 | (-0.458,+0.458,+0.428,-0.622) 1.6e-16 | (+1.648,-2.879,+1.047) 1.5e-16 | (-0.103,+0.395) 1.2e-16 |
| apolar_-.5 | 2 | (+0.000) 4.7e-04 | (-0.019,-0.482) 2.8e-01 | (+0.001,-0.001,+0.016,-0.000) 2.3e-03 | (+1.632,-2.799,+1.769) 4.7e-04 | (-0.500,+0.000) 2.4e-16 |
| apolar_-.5 | 4 | (+0.000) 7.9e-03 | (-0.043,-0.441) 4.0e-01 | (+0.001,-0.001,-0.001,-0.000) 8.0e-03 | (+2.188,-2.803,+1.530) 7.9e-03 | (-0.500,+0.000) 1.3e-15 |
| apolar_-.5 | 8 | (+0.000) 1.5e-02 | (-0.056,-0.382) 6.4e-01 | (+0.000,-0.001,+0.002,-0.000) 1.6e-02 | (+2.621,-5.070,+3.945) 2.1e-02 | (-0.500,-0.000) 4.3e-14 |
| apolar_-.5 | 16 | (-0.000) 2.7e-02 | (-0.061,-0.364) 8.2e-01 | (+0.001,-0.001,-0.002,-0.000) 2.7e-02 | (+3.027,-8.749,+9.121) 5.1e-01 | (-0.500,+0.000) 1.3e-12 |
| apolar_-.5 | 64 | (-0.000) 5.1e-02 | (-0.069,-0.314) 9.5e-01 | (+0.000,-0.001,-0.001,+0.000) 5.1e-02 | (+3.928,-7.361,+3.816) 3.6e-01 | (-0.500,+0.000) 2.0e-09 |
| cpolar(.3,-.4,-.2,.1) | 1 | (-0.314) 1.7e-01 | (-0.100,-0.610) 2.1e-16 | (-0.531,+0.431,-0.104,+0.406) 2.4e-16 | (+3.659,-7.953,+3.959) 1.7e-01 | (-0.600,-0.600) 1.7e-01 |
| cpolar(.3,-.4,-.2,.1) | 2 | (+0.443) 1.8e-01 | (-0.111,-0.186) 1.7e-01 | (+0.300,-0.400,-0.200,+0.100) 3.9e-16 | (+0.619,-1.951,+1.832) 1.9e-01 | (+1.051,+1.784) 2.5e-01 |
| cpolar(.3,-.4,-.2,.1) | 4 | (+0.287) 1.7e-01 | (-0.124,-0.202) 2.5e-01 | (+0.300,-0.400,-0.200,+0.100) 7.5e-16 | (+1.868,-1.165,+0.433) 1.7e-01 | (-0.439,+0.686) 2.2e-01 |
| cpolar(.3,-.4,-.2,.1) | 8 | (+0.108) 2.1e-01 | (-0.129,-0.279) 5.6e-01 | (+0.300,-0.400,-0.200,+0.100) 2.4e-14 | (+2.460,-3.645,+2.347) 2.1e-01 | (-0.298,-0.014) 2.4e-01 |
| cpolar(.3,-.4,-.2,.1) | 16 | (+0.048) 2.4e-01 | (-0.138,-0.285) 7.4e-01 | (+0.300,-0.400,-0.200,+0.100) 4.8e-13 | (+2.877,-6.945,+6.148) 2.7e-01 | (+0.110,+0.075) 2.6e-01 |
| cpolar(.3,-.4,-.2,.1) | 64 | (+0.034) 3.1e-01 | (-0.136,-0.287) 9.3e-01 | (+0.300,-0.400,-0.200,+0.100) 8.7e-11 | (+3.774,-8.725,+5.851) 4.5e-01 | (-0.017,+0.033) 3.1e-01 |

## generator = sketch_anisoA


### eps = 1e-06: fixed candidates confusable with the true rule

| true rule | r=1 | r=2 | r=4 | r=8 | r=16 | r=64 |
|---|---|---|---|---|---|---|
| frob | cpolar_colwhite, col_norm, polar, ns_cubic5, ns_cubic15, ns_quintic8, ns_quintic5, apolar_-.5, spec_p0.5, ns_quintic3, ns_quintic5_f32 | - | - | - | - | - |
| sign | cpolar_rowwhite, row_norm | - | - | - | - | - |
| row_norm | cpolar_rowwhite, sign | - | - | - | - | - |
| col_norm | frob, apolar_-.5, cpolar_colwhite, ns_quintic5, spec_p0.5, polar, ns_cubic5, ns_cubic15, ns_quintic3, ns_quintic8, ns_quintic5_f32 | - | - | - | - | - |
| polar | spec_p0.5, ns_quintic3, ns_quintic5, frob, col_norm, ns_cubic5, ns_cubic15, cpolar_colwhite, ns_quintic8, apolar_-.5, ns_quintic5_f32 | - | - | - | - | - |
| ns_quintic5 | polar, ns_cubic5, ns_cubic15, spec_p0.5, frob, col_norm, cpolar_colwhite, ns_quintic8, apolar_-.5, ns_quintic3, ns_quintic5_f32 | ns_quintic5_f32 | ns_quintic5_f32 | ns_quintic5_f32 | - | - |
| ns_quintic3 | ns_quintic5, frob, col_norm, apolar_-.5, cpolar_colwhite, ns_quintic8, spec_p0.5, polar, ns_cubic5, ns_cubic15, ns_quintic5_f32 | - | - | - | - | - |
| ns_quintic8 | polar, ns_cubic5, ns_cubic15, frob, col_norm, spec_p0.5, ns_quintic3, apolar_-.5, ns_quintic5, cpolar_colwhite, ns_quintic5_f32 | - | - | - | - | - |
| ns_cubic5 | polar, ns_cubic15, ns_quintic8, ns_quintic5, ns_quintic3, frob, col_norm, apolar_-.5, spec_p0.5, cpolar_colwhite, ns_quintic5_f32 | - | - | - | - | - |
| ns_cubic15 | polar, ns_cubic5, ns_quintic5, spec_p0.5, cpolar_colwhite, frob, col_norm, ns_quintic8, apolar_-.5, ns_quintic3, ns_quintic5_f32 | - | - | - | - | - |
| spec_p0.5 | polar, ns_cubic5, ns_cubic15, ns_quintic8, ns_quintic5, frob, col_norm, apolar_-.5, cpolar_colwhite, ns_quintic3, ns_quintic5_f32 | - | - | - | - | - |
| dpre_-.25_-.25 | cpolar_dshampoo | - | - | - | - | - |
| cpolar_rowwhite | sign, row_norm | - | - | - | - | - |
| cpolar_colwhite | apolar_-.5, polar, ns_cubic5, ns_cubic15, ns_quintic8, ns_quintic3, frob, col_norm, ns_quintic5, spec_p0.5, ns_quintic5_f32 | - | - | - | - | - |
| cpolar_dshampoo | dpre_-.25_-.25 | - | - | - | - | - |
| ns_quintic5_f32 | cpolar_colwhite, ns_quintic3, frob, col_norm, polar, ns_cubic5, ns_cubic15, ns_quintic5, apolar_-.5, ns_quintic8, spec_p0.5 | ns_quintic5 | ns_quintic5 | ns_quintic5 | - | - |
| ns_quintic5_bf16 | - | - | - | - | - | - |
| apolar_-.5 | frob, col_norm, cpolar_colwhite, ns_quintic3, ns_quintic8, polar, ns_cubic5, ns_cubic15, ns_quintic5, spec_p0.5, ns_quintic5_f32 | - | - | - | - | - |
| cpolar(.3,-.4,-.2,.1) | - | - | - | - | - | - |

### eps = 0.001: fixed candidates confusable with the true rule

| true rule | r=1 | r=2 | r=4 | r=8 | r=16 | r=64 |
|---|---|---|---|---|---|---|
| frob | cpolar_colwhite, col_norm, polar, ns_cubic5, ns_cubic15, ns_quintic8, ns_quintic5, apolar_-.5, spec_p0.5, ns_quintic3, ns_quintic5_f32 | - | - | - | - | - |
| sign | cpolar_rowwhite, row_norm | - | - | - | - | - |
| row_norm | cpolar_rowwhite, sign | - | - | - | - | - |
| col_norm | frob, apolar_-.5, cpolar_colwhite, ns_quintic5, spec_p0.5, polar, ns_cubic5, ns_cubic15, ns_quintic3, ns_quintic8, ns_quintic5_f32 | - | - | - | - | - |
| polar | spec_p0.5, ns_quintic3, ns_quintic5, frob, col_norm, ns_cubic5, ns_cubic15, cpolar_colwhite, ns_quintic8, apolar_-.5, ns_quintic5_f32 | ns_cubic15 | - | - | - | - |
| ns_quintic5 | polar, ns_cubic5, ns_cubic15, spec_p0.5, frob, col_norm, cpolar_colwhite, ns_quintic8, apolar_-.5, ns_quintic3, ns_quintic5_f32 | ns_quintic5_f32 | ns_quintic5_f32 | ns_quintic5_f32 | ns_quintic5_f32 | ns_quintic5_f32 |
| ns_quintic3 | ns_quintic5, frob, col_norm, apolar_-.5, cpolar_colwhite, ns_quintic8, spec_p0.5, polar, ns_cubic5, ns_cubic15, ns_quintic5_f32 | - | - | - | - | - |
| ns_quintic8 | polar, ns_cubic5, ns_cubic15, frob, col_norm, spec_p0.5, ns_quintic3, apolar_-.5, ns_quintic5, cpolar_colwhite, ns_quintic5_f32 | - | - | - | - | - |
| ns_cubic5 | polar, ns_cubic15, ns_quintic8, ns_quintic5, ns_quintic3, frob, col_norm, apolar_-.5, spec_p0.5, cpolar_colwhite, ns_quintic5_f32 | - | - | - | - | - |
| ns_cubic15 | polar, ns_cubic5, ns_quintic5, spec_p0.5, cpolar_colwhite, frob, col_norm, ns_quintic8, apolar_-.5, ns_quintic3, ns_quintic5_f32 | - | polar | - | - | - |
| spec_p0.5 | polar, ns_cubic5, ns_cubic15, ns_quintic8, ns_quintic5, frob, col_norm, apolar_-.5, cpolar_colwhite, ns_quintic3, ns_quintic5_f32 | - | - | - | - | - |
| dpre_-.25_-.25 | cpolar_dshampoo | - | - | - | - | - |
| cpolar_rowwhite | sign, row_norm | - | - | - | - | - |
| cpolar_colwhite | apolar_-.5, polar, ns_cubic5, ns_cubic15, ns_quintic8, ns_quintic3, frob, col_norm, ns_quintic5, spec_p0.5, ns_quintic5_f32 | - | - | - | - | - |
| cpolar_dshampoo | dpre_-.25_-.25 | - | - | - | - | - |
| ns_quintic5_f32 | cpolar_colwhite, ns_quintic3, frob, col_norm, polar, ns_cubic5, ns_cubic15, ns_quintic5, apolar_-.5, ns_quintic8, spec_p0.5 | ns_quintic5 | ns_quintic5 | ns_quintic5 | ns_quintic5 | ns_quintic5 |
| ns_quintic5_bf16 | - | - | - | - | - | - |
| apolar_-.5 | frob, col_norm, cpolar_colwhite, ns_quintic3, ns_quintic8, polar, ns_cubic5, ns_cubic15, ns_quintic5, spec_p0.5, ns_quintic5_f32 | - | - | - | - | - |
| cpolar(.3,-.4,-.2,.1) | - | - | - | - | - | - |

### eps = 0.01: fixed candidates confusable with the true rule

| true rule | r=1 | r=2 | r=4 | r=8 | r=16 | r=64 |
|---|---|---|---|---|---|---|
| frob | cpolar_colwhite, col_norm, polar, ns_cubic5, ns_cubic15, ns_quintic8, ns_quintic5, apolar_-.5, spec_p0.5, ns_quintic3, ns_quintic5_f32 | - | - | - | - | - |
| sign | cpolar_rowwhite, row_norm | - | - | - | - | - |
| row_norm | cpolar_rowwhite, sign | - | - | - | - | - |
| col_norm | frob, apolar_-.5, cpolar_colwhite, ns_quintic5, spec_p0.5, polar, ns_cubic5, ns_cubic15, ns_quintic3, ns_quintic8, ns_quintic5_f32 | - | - | - | - | - |
| polar | spec_p0.5, ns_quintic3, ns_quintic5, frob, col_norm, ns_cubic5, ns_cubic15, cpolar_colwhite, ns_quintic8, apolar_-.5, ns_quintic5_f32 | ns_cubic15 | ns_cubic15 | - | - | - |
| ns_quintic5 | polar, ns_cubic5, ns_cubic15, spec_p0.5, frob, col_norm, cpolar_colwhite, ns_quintic8, apolar_-.5, ns_quintic3, ns_quintic5_f32 | ns_quintic5_f32 | ns_quintic5_f32 | ns_quintic5_f32 | ns_quintic5_f32 | ns_quintic5_f32 |
| ns_quintic3 | ns_quintic5, frob, col_norm, apolar_-.5, cpolar_colwhite, ns_quintic8, spec_p0.5, polar, ns_cubic5, ns_cubic15, ns_quintic5_f32 | - | - | - | - | - |
| ns_quintic8 | polar, ns_cubic5, ns_cubic15, frob, col_norm, spec_p0.5, ns_quintic3, apolar_-.5, ns_quintic5, cpolar_colwhite, ns_quintic5_f32 | - | - | - | - | - |
| ns_cubic5 | polar, ns_cubic15, ns_quintic8, ns_quintic5, ns_quintic3, frob, col_norm, apolar_-.5, spec_p0.5, cpolar_colwhite, ns_quintic5_f32 | - | - | - | - | - |
| ns_cubic15 | polar, ns_cubic5, ns_quintic5, spec_p0.5, cpolar_colwhite, frob, col_norm, ns_quintic8, apolar_-.5, ns_quintic3, ns_quintic5_f32 | polar | polar | - | - | - |
| spec_p0.5 | polar, ns_cubic5, ns_cubic15, ns_quintic8, ns_quintic5, frob, col_norm, apolar_-.5, cpolar_colwhite, ns_quintic3, ns_quintic5_f32 | - | - | - | - | - |
| dpre_-.25_-.25 | cpolar_dshampoo | - | - | - | - | - |
| cpolar_rowwhite | sign, row_norm | - | - | - | - | - |
| cpolar_colwhite | apolar_-.5, polar, ns_cubic5, ns_cubic15, ns_quintic8, ns_quintic3, frob, col_norm, ns_quintic5, spec_p0.5, ns_quintic5_f32 | - | - | - | - | - |
| cpolar_dshampoo | dpre_-.25_-.25 | - | - | - | - | - |
| ns_quintic5_f32 | cpolar_colwhite, ns_quintic3, frob, col_norm, polar, ns_cubic5, ns_cubic15, ns_quintic5, apolar_-.5, ns_quintic8, spec_p0.5 | ns_quintic5 | ns_quintic5 | ns_quintic5 | ns_quintic5 | ns_quintic5 |
| ns_quintic5_bf16 | - | - | - | - | - | - |
| apolar_-.5 | frob, col_norm, cpolar_colwhite, ns_quintic3, ns_quintic8, polar, ns_cubic5, ns_cubic15, ns_quintic5, spec_p0.5, ns_quintic5_f32 | - | - | - | - | - |
| cpolar(.3,-.4,-.2,.1) | - | - | - | - | - | - |

### noise-free misfit to the NEAREST other fixed candidate (log10)

| true rule | r=1 | r=2 | r=4 | r=8 | r=16 | r=64 |
|---|---|---|---|---|---|---|
| frob | -16.1 (cpolar_colwhite) | -1.0 (ns_cubic5) | -0.6 (spec_p0.5) | -0.5 (spec_p0.5) | -0.5 (spec_p0.5) | -0.5 (spec_p0.5) |
| sign | -15.4 (cpolar_rowwhite) | -0.2 (row_norm) | -0.1 (dpre_-.25_-.25) | -0.1 (dpre_-.25_-.25) | -0.1 (dpre_-.25_-.25) | -0.1 (dpre_-.25_-.25) |
| row_norm | -15.4 (cpolar_rowwhite) | -0.3 (dpre_-.25_-.25) | -0.3 (dpre_-.25_-.25) | -0.3 (dpre_-.25_-.25) | -0.3 (dpre_-.25_-.25) | -0.3 (dpre_-.25_-.25) |
| col_norm | -15.9 (frob) | -0.6 (cpolar_colwhite) | -0.3 (cpolar_colwhite) | -0.2 (cpolar_colwhite) | -0.2 (dpre_-.25_-.25) | -0.2 (dpre_-.25_-.25) |
| polar | -16.0 (spec_p0.5) | -3.1 (ns_cubic15) | -2.4 (ns_cubic15) | -0.7 (ns_quintic8) | -0.8 (ns_quintic8) | -0.6 (ns_quintic8) |
| ns_quintic5 | -15.9 (ns_cubic15) | -6.4 (ns_quintic5_f32) | -6.3 (ns_quintic5_f32) | -6.2 (ns_quintic5_f32) | -5.8 (ns_quintic5_f32) | -5.2 (ns_quintic5_f32) |
| ns_quintic3 | -15.9 (ns_quintic5) | -0.5 (ns_quintic8) | -0.4 (ns_cubic15) | -0.4 (ns_cubic15) | -0.4 (ns_cubic15) | -0.4 (ns_cubic5) |
| ns_quintic8 | -15.8 (ns_cubic15) | -0.8 (ns_cubic15) | -0.8 (ns_cubic15) | -0.7 (polar) | -0.8 (polar) | -0.7 (polar) |
| ns_cubic5 | -15.9 (ns_cubic15) | -1.6 (spec_p0.5) | -0.5 (spec_p0.5) | -0.5 (spec_p0.5) | -0.6 (spec_p0.5) | -0.6 (spec_p0.5) |
| ns_cubic15 | -16.4 (ns_cubic5) | -2.0 (polar) | -4.1 (polar) | -0.8 (polar) | -0.8 (ns_quintic5) | -0.8 (ns_quintic5_f32) |
| spec_p0.5 | -16.1 (ns_cubic15) | -1.5 (ns_cubic5) | -0.6 (frob) | -0.5 (ns_cubic5) | -0.6 (ns_cubic5) | -0.6 (ns_cubic5) |
| dpre_-.25_-.25 | -15.8 (cpolar_dshampoo) | -0.3 (spec_p0.5) | -0.3 (spec_p0.5) | -0.3 (spec_p0.5) | -0.3 (spec_p0.5) | -0.3 (row_norm) |
| cpolar_rowwhite | -15.6 (row_norm) | -0.2 (cpolar_dshampoo) | -0.3 (cpolar_dshampoo) | -0.3 (cpolar_dshampoo) | -0.3 (cpolar_dshampoo) | -0.3 (cpolar_dshampoo) |
| cpolar_colwhite | -15.8 (apolar_-.5) | -1.2 (apolar_-.5) | -0.9 (apolar_-.5) | -0.8 (apolar_-.5) | -0.6 (apolar_-.5) | -0.6 (apolar_-.5) |
| cpolar_dshampoo | -15.6 (dpre_-.25_-.25) | -0.2 (ns_cubic15) | -0.3 (ns_cubic15) | -0.3 (polar) | -0.3 (polar) | -0.3 (polar) |
| ns_quintic5_f32 | -6.5 (cpolar_colwhite) | -6.4 (ns_quintic5) | -6.2 (ns_quintic5) | -6.0 (ns_quintic5) | -5.8 (ns_quintic5) | -5.2 (ns_quintic5) |
| ns_quintic5_bf16 | -1.7 (apolar_-.5) | -1.7 (ns_quintic5_f32) | -1.6 (ns_quintic5) | -1.5 (ns_quintic5) | -1.2 (ns_quintic5_f32) | -0.8 (ns_quintic5) |
| apolar_-.5 | -16.0 (col_norm) | -1.1 (cpolar_colwhite) | -1.0 (cpolar_colwhite) | -0.8 (cpolar_colwhite) | -0.6 (cpolar_colwhite) | -0.6 (cpolar_colwhite) |
| cpolar(.3,-.4,-.2,.1) | -0.8 (ns_quintic5_f32) | -0.7 (spec_p0.5) | -0.7 (spec_p0.5) | -0.4 (spec_p0.5) | -0.4 (ns_cubic15) | -0.4 (cpolar_dshampoo) |

### noise-free family fits (fitted params; held-out misfit)

| true rule | r | FIT spec_power | FIT diag_precond | FIT cpolar | FIT ns5_coeffs | FIT apolar |
|---|---|---|---|---|---|---|
| frob | 1 | (+1.182) 2.1e-16 | (-0.000,-1.012) 1.3e-16 | (-0.458,+0.458,-0.620,-0.622) 2.0e-16 | (+1.551,-1.609,+1.141) 3.4e-16 | (+0.391,-0.610) 2.0e-16 |
| frob | 2 | (+1.000) 3.2e-09 | (+0.000,-0.000) 2.4e-16 | (+0.214,-0.214,+0.296,+0.439) 9.8e-03 | (+1.342,-0.457,+0.220) 1.1e-08 | (+0.489,+0.489) 1.2e-02 |
| frob | 4 | (+1.000) 3.2e-09 | (+0.000,-0.000) 4.2e-16 | (+0.805,-0.805,-0.313,+0.159) 5.7e-02 | (+1.018,-0.000,+0.000) 4.1e-09 | (+0.462,+0.462) 1.3e-01 |
| frob | 8 | (+1.000) 1.3e-09 | (-0.000,-0.000) 4.7e-16 | (+0.814,-0.812,-0.165,+0.235) 1.5e-01 | (+0.661,+0.000,-0.000) 1.8e-08 | (+0.585,+0.585) 3.3e-01 |
| frob | 16 | (+1.000) 4.5e-09 | (-0.000,+0.000) 3.0e-16 | (+0.957,-0.946,+0.033,+0.358) 3.3e-01 | (+0.663,+0.000,-0.000) 8.4e-09 | (+0.662,+0.662) 5.8e-01 |
| frob | 64 | (+1.000) 2.9e-09 | (+0.000,-0.000) 9.6e-16 | (+0.938,-0.889,+0.401,+0.650) 6.2e-01 | (+0.301,+0.000,-0.000) 5.2e-09 | (+1.015,+1.015) 7.7e-01 |
| row_norm | 1 | (-0.300) 8.2e-01 | (-0.500,-0.104) 1.5e-16 | (-0.232,-0.268,+0.357,+0.119) 8.1e-16 | (+2.949,-7.193,+3.635) 8.2e-01 | (-0.600,-0.600) 8.2e-01 |
| row_norm | 2 | (+0.626) 8.2e-01 | (-0.500,+0.000) 1.8e-16 | (+0.018,-0.518,+0.210,+0.237) 1.8e-02 | (+2.548,-0.784,+0.106) 8.2e-01 | (+0.027,+0.327) 8.2e-01 |
| row_norm | 4 | (+0.608) 7.9e-01 | (-0.500,+0.000) 2.0e-16 | (+0.233,-0.732,-0.038,+0.201) 1.1e-01 | (+1.546,-0.563,+0.185) 7.9e-01 | (+0.297,+0.333) 8.0e-01 |
| row_norm | 8 | (+0.667) 7.6e-01 | (-0.500,-0.000) 2.5e-16 | (+0.320,-0.813,+0.032,+0.285) 2.0e-01 | (+1.446,-0.114,+0.012) 7.6e-01 | (+0.209,+0.363) 7.8e-01 |
| row_norm | 16 | (+0.720) 7.3e-01 | (-0.500,-0.000) 2.3e-16 | (+0.476,-0.959,+0.216,+0.403) 3.6e-01 | (+1.261,-0.138,+0.028) 7.3e-01 | (+0.421,+0.529) 7.9e-01 |
| row_norm | 64 | (+0.771) 7.3e-01 | (-0.500,+0.000) 2.1e-16 | (+0.468,-0.929,+0.615,+0.718) 6.9e-01 | (+1.144,-0.163,+0.034) 7.2e-01 | (+1.112,+1.171) 8.9e-01 |
| col_norm | 1 | (-0.300) 1.9e-16 | (+0.000,-0.151) 1.3e-16 | (-0.458,+0.458,-0.622,-0.621) 1.7e-16 | (+1.859,-6.942,+3.815) 1.9e-16 | (-0.617,-0.098) 1.5e-16 |
| col_norm | 2 | (+0.038) 3.5e-01 | (+0.000,-0.500) 2.0e-16 | (+0.542,-0.543,-0.460,-0.268) 9.6e-02 | (+3.384,-4.067,+1.755) 3.7e-01 | (-0.517,-0.000) 2.6e-01 |
| col_norm | 4 | (+0.118) 5.3e-01 | (+0.000,-0.500) 1.5e-16 | (+0.481,-0.488,-0.601,-0.120) 2.6e-01 | (+3.301,-0.795,+0.057) 5.4e-01 | (-0.500,-0.003) 4.7e-01 |
| col_norm | 8 | (+0.239) 6.5e-01 | (-0.000,-0.500) 2.4e-16 | (+0.686,-0.698,-0.522,-0.082) 5.0e-01 | (+2.643,-0.554,+0.048) 6.5e-01 | (-0.512,+0.010) 6.5e-01 |
| col_norm | 16 | (+0.463) 7.6e-01 | (+0.000,-0.500) 1.8e-16 | (+0.923,-0.871,-0.536,-0.026) 6.7e-01 | (+2.014,-0.436,+0.055) 7.6e-01 | (-0.461,+0.018) 8.2e-01 |
| col_norm | 64 | (+0.685) 7.9e-01 | (+0.000,-0.500) 1.9e-16 | (+0.814,-0.727,-0.507,+0.000) 9.0e-01 | (+1.537,-0.151,+0.019) 7.9e-01 | (-0.457,+0.027) 9.5e-01 |
| polar | 1 | (-0.307) 1.7e-16 | (-0.000,+0.642) 2.5e-16 | (+0.458,-0.458,+0.415,+0.428) 3.0e-16 | (+1.575,-1.660,+1.149) 1.4e-16 | (-0.106,-0.093) 1.6e-16 |
| polar | 2 | (-0.000) 2.8e-09 | (-0.031,-0.485) 2.8e-01 | (+0.000,-0.000,+0.000,-0.000) 3.8e-16 | (+2.867,-4.649,+2.779) 2.9e-02 | (-0.000,-0.000) 3.8e-16 |
| polar | 4 | (-0.000) 1.2e-08 | (-0.053,-0.450) 5.4e-01 | (-0.000,+0.000,+0.000,+0.000) 2.4e-15 | (+3.396,-6.449,+3.980) 7.4e-02 | (-0.000,+0.000) 2.4e-15 |
| polar | 8 | (+0.000) 5.2e-09 | (-0.060,-0.385) 7.1e-01 | (+0.000,+0.000,+0.000,-0.000) 8.1e-14 | (+3.432,-7.548,+5.312) 5.0e-01 | (-0.000,+0.000) 1.1e-13 |
| polar | 16 | (+0.000) 3.0e-10 | (-0.069,-0.301) 8.4e-01 | (-0.000,+0.000,-0.000,+0.000) 8.0e-13 | (+3.643,-7.558,+4.711) 3.8e-01 | (+0.000,+0.000) 1.4e-12 |
| polar | 64 | (+0.000) 1.2e-09 | (-0.071,-0.179) 9.6e-01 | (-0.000,+0.000,+0.000,-0.000) 3.7e-07 | (+4.106,-7.516,+3.603) 6.3e-01 | (-0.000,+0.000) 5.3e-07 |
| ns_quintic5 | 1 | (-0.306) 3.2e-16 | (-0.000,-0.930) 1.5e-16 | (-0.458,+0.458,-0.622,+0.414) 2.0e-16 | (+1.725,-5.907,+3.972) 2.1e-16 | (+0.394,-0.103) 2.3e-16 |
| ns_quintic5 | 2 | (-0.063) 8.5e-02 | (-0.031,-0.518) 3.3e-01 | (+0.100,-0.101,+0.305,-0.060) 9.8e-02 | (+3.445,-4.775,+2.031) 2.1e-09 | (+0.003,-0.032) 9.0e-02 |
| ns_quintic5 | 4 | (-0.051) 1.3e-01 | (-0.072,-0.467) 5.7e-01 | (+0.022,-0.021,+0.008,-0.030) 1.4e-01 | (+3.445,-4.775,+2.032) 4.6e-09 | (+0.002,-0.028) 1.3e-01 |
| ns_quintic5 | 8 | (+0.029) 2.3e-01 | (-0.068,-0.387) 7.0e-01 | (-0.046,+0.044,-0.000,+0.022) 2.4e-01 | (+3.445,-4.775,+2.032) 5.9e-09 | (+0.001,+0.015) 2.4e-01 |
| ns_quintic5 | 16 | (+0.117) 3.2e-01 | (-0.058,-0.274) 7.7e-01 | (-0.009,+0.009,+0.016,+0.107) 3.4e-01 | (+3.444,-4.775,+2.032) 7.2e-09 | (+0.014,+0.111) 3.3e-01 |
| ns_quintic5 | 64 | (+0.181) 4.7e-01 | (-0.060,-0.139) 8.7e-01 | (+0.047,-0.043,+0.047,+0.242) 3.8e-01 | (+3.445,-4.775,+2.032) 4.9e-09 | (+0.054,+0.272) 3.7e-01 |
| spec_p0.5 | 1 | (+0.400) 1.9e-16 | (-0.000,+0.524) 2.1e-16 | (-0.458,+0.458,+0.414,-0.622) 2.7e-16 | (+2.359,-5.929,+1.422) 9.8e-17 | (-0.105,-0.100) 1.7e-16 |
| spec_p0.5 | 2 | (+0.500) 3.8e-09 | (-0.001,-0.233) 4.0e-02 | (-0.027,+0.027,+0.199,+0.263) 1.0e-02 | (+0.189,-55.708,+55.571) 6.1e-03 | (+0.186,+0.250) 2.7e-02 |
| spec_p0.5 | 4 | (+0.500) 3.9e-09 | (-0.013,-0.216) 1.4e-01 | (+0.263,-0.264,+0.008,+0.181) 7.4e-02 | (+1.680,-0.930,+0.366) 4.1e-02 | (+0.144,+0.259) 1.3e-01 |
| spec_p0.5 | 8 | (+0.500) 1.7e-09 | (-0.020,-0.179) 2.0e-01 | (+0.495,-0.500,-0.048,+0.157) 1.7e-01 | (+1.809,-0.336,+0.043) 9.3e-02 | (+0.186,+0.317) 2.3e-01 |
| spec_p0.5 | 16 | (+0.500) 5.1e-09 | (-0.025,-0.143) 2.7e-01 | (+0.607,-0.615,+0.041,+0.222) 2.8e-01 | (+1.727,-0.320,+0.046) 1.0e-01 | (+0.260,+0.414) 4.2e-01 |
| spec_p0.5 | 64 | (+0.500) 5.5e-09 | (-0.025,-0.058) 3.0e-01 | (+0.781,-0.778,+0.200,+0.468) 5.6e-01 | (+1.916,-0.230,+0.019) 1.4e-01 | (+0.483,+0.688) 6.6e-01 |
| dpre_-.25_-.25 | 1 | (+1.157) 4.4e-01 | (-0.250,-0.003) 2.2e-16 | (-0.125,-0.125,-0.461,-0.079) 1.9e-16 | (+4.331,-2.635,+0.591) 4.4e-01 | (-0.600,-0.600) 4.4e-01 |
| dpre_-.25_-.25 | 2 | (+0.376) 4.8e-01 | (-0.250,-0.250) 3.0e-16 | (+0.396,-0.645,-0.543,-0.096) 4.5e-02 | (+0.163,-14.107,+14.049) 4.8e-01 | (-0.071,+0.182) 4.8e-01 |
| dpre_-.25_-.25 | 4 | (+0.384) 4.5e-01 | (-0.250,-0.250) 3.4e-16 | (+0.376,-0.628,-0.389,+0.026) 1.8e-01 | (+1.795,-0.933,+0.308) 4.5e-01 | (-0.035,+0.193) 4.8e-01 |
| dpre_-.25_-.25 | 8 | (+0.472) 4.9e-01 | (-0.250,-0.250) 2.7e-16 | (+0.533,-0.778,-0.326,+0.086) 3.5e-01 | (+0.781,-2.168,+2.250) 5.0e-01 | (+0.061,+0.280) 5.9e-01 |
| dpre_-.25_-.25 | 16 | (+0.582) 5.0e-01 | (-0.250,-0.250) 3.8e-16 | (+0.672,-0.894,-0.174,+0.206) 5.2e-01 | (+1.534,-0.219,+0.033) 5.0e-01 | (+0.169,+0.368) 7.2e-01 |
| dpre_-.25_-.25 | 64 | (+0.708) 5.1e-01 | (-0.250,-0.250) 8.1e-16 | (+0.675,-0.864,+0.106,+0.465) 7.9e-01 | (+1.271,-0.232,+0.075) 5.1e-01 | (+0.586,+0.804) 8.7e-01 |
| cpolar_rowwhite | 1 | (-0.300) 8.1e-01 | (-0.500,+0.417) 2.2e-16 | (-0.754,+0.254,-0.644,+0.449) 3.1e-16 | (+2.000,-1.500,+0.500) 8.1e-01 | (-0.100,-0.100) 8.1e-01 |
| cpolar_rowwhite | 2 | (+0.145) 9.0e-01 | (-0.501,-0.255) 1.1e-01 | (+0.001,-0.501,+1.182,-0.001) 3.5e-01 | (+1.223,-86.377,+85.030) 9.0e-01 | (+1.172,+0.073) 9.0e-01 |
| cpolar_rowwhite | 4 | (+0.065) 8.0e-01 | (-0.526,-0.344) 4.2e-01 | (-0.000,-0.500,-0.000,+0.000) 9.7e-14 | (+2.708,-3.167,+1.490) 8.1e-01 | (-0.019,+0.032) 8.0e-01 |
| cpolar_rowwhite | 8 | (+0.021) 7.9e-01 | (-0.548,-0.344) 6.0e-01 | (+0.000,-0.500,-0.000,-0.000) 1.7e-13 | (+3.391,-7.140,+4.853) 8.0e-01 | (-0.010,+0.014) 7.9e-01 |
| cpolar_rowwhite | 16 | (+0.003) 7.5e-01 | (-0.549,-0.269) 8.0e-01 | (+0.000,-0.500,+0.000,+0.000) 1.4e-12 | (+3.665,-9.099,+6.727) 7.9e-01 | (-0.011,+0.003) 7.5e-01 |
| cpolar_rowwhite | 64 | (-0.005) 7.1e-01 | (-0.543,-0.159) 9.4e-01 | (-0.000,-0.500,+0.000,+0.000) 1.0e-11 | (+4.088,-8.626,+4.780) 8.3e-01 | (-0.019,-0.004) 7.1e-01 |
| cpolar_colwhite | 1 | (-0.300) 2.0e-16 | (-0.000,-0.928) 1.6e-16 | (-0.458,+0.458,-0.620,-0.104) 1.9e-16 | (+3.546,-6.657,+1.564) 2.6e-16 | (-0.616,-0.100) 1.3e-16 |
| cpolar_colwhite | 2 | (+0.000) 3.0e-01 | (-0.016,-0.497) 3.1e-01 | (+0.000,-0.000,-0.500,-0.000) 1.9e-16 | (+3.072,-5.080,+3.006) 3.0e-01 | (-0.511,-0.000) 6.5e-02 |
| cpolar_colwhite | 4 | (+0.000) 3.2e-01 | (-0.034,-0.501) 3.9e-01 | (+0.000,-0.000,-0.500,-0.000) 1.6e-15 | (+3.063,-5.361,+3.306) 3.2e-01 | (-0.488,-0.000) 1.2e-01 |
| cpolar_colwhite | 8 | (+0.010) 5.0e-01 | (-0.038,-0.500) 6.4e-01 | (+0.000,-0.000,-0.500,-0.000) 1.1e-13 | (+3.385,-6.077,+3.565) 5.4e-01 | (-0.492,-0.000) 1.6e-01 |
| cpolar_colwhite | 16 | (-0.003) 5.9e-01 | (-0.050,-0.502) 8.0e-01 | (-0.000,+0.000,-0.500,-0.000) 1.7e-12 | (+3.652,-8.140,+5.428) 6.7e-01 | (-0.482,-0.001) 2.4e-01 |
| cpolar_colwhite | 64 | (-0.027) 5.9e-01 | (-0.052,-0.521) 9.4e-01 | (-0.000,+0.000,-0.500,+0.000) 7.7e-10 | (+4.108,-7.310,+3.392) 7.8e-01 | (-0.502,+0.001) 2.4e-01 |
| cpolar_dshampoo | 1 | (-0.304) 4.6e-01 | (-0.250,-0.146) 1.1e-16 | (-0.125,-0.125,-0.477,-0.461) 1.8e-16 | (+1.833,-3.240,+2.849) 4.6e-01 | (+0.400,-0.100) 4.6e-01 |
| cpolar_dshampoo | 2 | (+0.042) 5.9e-01 | (-0.261,-0.415) 2.8e-01 | (+0.000,-0.250,-0.250,-0.000) 1.8e-16 | (+2.969,-4.677,+2.710) 6.0e-01 | (-0.259,+0.020) 5.9e-01 |
| cpolar_dshampoo | 4 | (+0.018) 4.9e-01 | (-0.279,-0.425) 4.1e-01 | (+0.000,-0.250,-0.250,-0.000) 2.5e-16 | (+3.250,-6.092,+3.838) 5.0e-01 | (-0.240,+0.009) 4.9e-01 |
| cpolar_dshampoo | 8 | (+0.007) 4.8e-01 | (-0.295,-0.409) 6.1e-01 | (+0.000,-0.250,-0.250,-0.000) 2.1e-14 | (+3.397,-7.577,+5.460) 5.1e-01 | (-0.208,+0.003) 4.5e-01 |
| cpolar_dshampoo | 16 | (+0.001) 5.1e-01 | (-0.303,-0.362) 8.0e-01 | (+0.000,-0.250,-0.250,-0.000) 7.2e-13 | (+3.873,-6.839,+3.383) 6.4e-01 | (-0.238,+0.001) 4.7e-01 |
| cpolar_dshampoo | 64 | (-0.005) 4.6e-01 | (-0.287,-0.306) 9.5e-01 | (-0.000,-0.250,-0.250,+0.000) 1.7e-09 | (+4.095,-7.202,+3.320) 7.3e-01 | (-0.254,-0.001) 4.1e-01 |
| apolar_-.5 | 1 | (-0.300) 1.5e-16 | (+0.000,-0.712) 1.2e-16 | (-0.458,+0.458,+0.428,-0.622) 1.6e-16 | (+1.648,-2.879,+1.047) 1.5e-16 | (-0.103,+0.395) 1.2e-16 |
| apolar_-.5 | 2 | (+0.000) 2.7e-01 | (-0.021,-0.496) 2.8e-01 | (-0.001,+0.001,-0.493,+0.000) 7.6e-02 | (+3.238,-4.376,+2.145) 2.8e-01 | (-0.500,+0.000) 1.8e-16 |
| apolar_-.5 | 4 | (+0.001) 3.3e-01 | (-0.039,-0.494) 4.0e-01 | (-0.001,+0.001,-0.474,-0.000) 1.1e-01 | (+3.013,-5.478,+3.480) 3.4e-01 | (-0.500,+0.000) 3.8e-16 |
| apolar_-.5 | 8 | (+0.011) 5.2e-01 | (-0.050,-0.484) 6.5e-01 | (+0.002,-0.003,-0.459,+0.000) 1.9e-01 | (+3.387,-7.036,+4.759) 5.5e-01 | (-0.500,-0.000) 1.0e-13 |
| apolar_-.5 | 16 | (-0.006) 6.0e-01 | (-0.058,-0.494) 8.2e-01 | (-0.005,+0.005,-0.460,+0.001) 2.4e-01 | (+3.674,-7.596,+4.476) 7.0e-01 | (-0.500,+0.000) 1.2e-12 |
| apolar_-.5 | 64 | (-0.023) 5.4e-01 | (-0.063,-0.480) 9.5e-01 | (+0.001,-0.002,-0.448,-0.000) 2.3e-01 | (+4.121,-7.629,+3.648) 7.7e-01 | (-0.500,-0.000) 6.3e-10 |
| cpolar(.3,-.4,-.2,.1) | 1 | (-0.314) 1.7e-01 | (-0.100,-0.610) 2.1e-16 | (-0.531,+0.431,-0.104,+0.406) 2.4e-16 | (+3.659,-7.953,+3.959) 1.7e-01 | (-0.600,-0.600) 1.7e-01 |
| cpolar(.3,-.4,-.2,.1) | 2 | (+0.507) 1.8e-01 | (-0.101,-0.202) 2.9e-02 | (+0.300,-0.400,-0.200,+0.100) 2.2e-16 | (+0.709,-7.058,+7.303) 1.8e-01 | (+0.036,+0.252) 1.8e-01 |
| cpolar(.3,-.4,-.2,.1) | 4 | (+0.424) 2.0e-01 | (-0.108,-0.210) 1.6e-01 | (+0.300,-0.400,-0.200,+0.100) 1.0e-15 | (+2.860,-0.596,+0.050) 2.0e-01 | (-0.050,+0.238) 2.2e-01 |
| cpolar(.3,-.4,-.2,.1) | 8 | (+0.247) 2.7e-01 | (-0.119,-0.230) 4.1e-01 | (+0.300,-0.400,-0.200,+0.100) 2.6e-14 | (+2.531,-2.854,+1.378) 3.1e-01 | (-0.138,+0.161) 2.7e-01 |
| cpolar(.3,-.4,-.2,.1) | 16 | (+0.142) 3.2e-01 | (-0.136,-0.219) 6.4e-01 | (+0.300,-0.400,-0.200,+0.100) 2.6e-13 | (+3.466,-4.010,+1.431) 4.3e-01 | (-0.181,+0.137) 2.8e-01 |
| cpolar(.3,-.4,-.2,.1) | 64 | (+0.085) 3.7e-01 | (-0.136,-0.168) 8.9e-01 | (+0.300,-0.400,-0.200,+0.100) 1.6e-10 | (+3.811,-7.939,+4.740) 5.7e-01 | (-0.192,+0.112) 3.1e-01 |

## Noisy verification (n=12 modules, 8 train / 4 test, noise on D)

rank = position of the true rule among fixed candidates by held-out error; nearest = closest rival as err_C/err_T; recovery = fitted family parameters.

| kind | true rule | r | eps | err_T | rank | nearest rival (ratio) | recovered (true) |
|---|---|---|---|---|---|---|---|
| parametric | frob | 1 | 1e-06 | 1.02e-06 | 2 | cpolar_colwhite (1) | (-0.292) (+1) |
| parametric | sign | 1 | 1e-06 | 1.00e-06 | 2 | cpolar_rowwhite (1) |  |
| parametric | row_norm | 1 | 1e-06 | 1.00e-06 | 3 | cpolar_rowwhite (1) | (-0.500,-0.625) (-0.5,+0) |
| parametric | col_norm | 1 | 1e-06 | 1.00e-06 | 2 | frob (1) | (-0.000,+0.745) (+0,-0.5) |
| parametric | polar | 1 | 1e-06 | 1.01e-06 | 5 | frob (1) | (+0.469) (+0) |
| parametric | ns_quintic5 | 1 | 1e-06 | 1.01e-06 | 4 | cpolar_colwhite (1) | (+1.906,-4.102,+0.917) (+3.44,-4.78,+2.03) |
| parametric | ns_quintic3 | 1 | 1e-06 | 1.01e-06 | 4 | spec_p0.5 (1) |  |
| parametric | ns_quintic8 | 1 | 1e-06 | 1.01e-06 | 4 | ns_cubic15 (1) |  |
| parametric | ns_cubic5 | 1 | 1e-06 | 1.01e-06 | 4 | ns_quintic5 (1) |  |
| parametric | ns_cubic15 | 1 | 1e-06 | 9.92e-07 | 5 | ns_quintic5 (1) |  |
| parametric | spec_p0.5 | 1 | 1e-06 | 1.01e-06 | 1 | cpolar_colwhite (1) | (+1.045) (+0.5) |
| parametric | dpre_-.25_-.25 | 1 | 1e-06 | 9.96e-07 | 2 | cpolar_dshampoo (1) | (-0.250,+0.121) (-0.25,-0.25) |
| parametric | cpolar_rowwhite | 1 | 1e-06 | 1.01e-06 | 1 | row_norm (1) | (-0.754,+0.254,-0.673,-0.107) (+0,-0.5,+0,+0) |
| parametric | cpolar_colwhite | 1 | 1e-06 | 1.00e-06 | 10 | ns_quintic8 (1) | (+0.456,-0.456,-0.107,-0.623) (+0,+0,-0.5,+0) |
| parametric | cpolar_dshampoo | 1 | 1e-06 | 9.96e-07 | 2 | dpre_-.25_-.25 (1) | (-0.125,-0.125,-0.461,-0.079) (+0,-0.25,-0.25,+0) |
| parametric | ns_quintic5_f32 | 1 | 1e-06 | 1.02e-06 | 1 | ns_quintic8 (1.04) |  |
| parametric | ns_quintic5_bf16 | 1 | 1e-06 | 1.01e-06 | 1 | col_norm (1.74e+04) |  |
| parametric | cpolar(.3,-.4,-.2,.1) | 1 | 1e-06 | nan | None | frob (nan) | (-0.532,+0.432,-0.625,-0.606) (+0.3,-0.4,-0.2,+0.1) |
| parametric | frob | 1 | 0.001 | 1.02e-03 | 2 | ns_quintic5_f32 (1) | (+1.100) (+1) |
| parametric | sign | 1 | 0.001 | 1.00e-03 | 2 | cpolar_rowwhite (1) |  |
| parametric | row_norm | 1 | 0.001 | 1.00e-03 | 3 | cpolar_rowwhite (1) | (-0.500,-0.623) (-0.5,+0) |
| parametric | col_norm | 1 | 0.001 | 1.00e-03 | 3 | ns_quintic5_f32 (1) | (-0.000,-0.156) (+0,-0.5) |
| parametric | polar | 1 | 0.001 | 1.01e-03 | 6 | frob (1) | (-0.315) (+0) |
| parametric | ns_quintic5 | 1 | 0.001 | 1.01e-03 | 4 | cpolar_colwhite (1) | (+4.599,-7.713,+2.893) (+3.44,-4.78,+2.03) |
| parametric | ns_quintic3 | 1 | 0.001 | 1.01e-03 | 6 | ns_quintic5_f32 (1) |  |
| parametric | ns_quintic8 | 1 | 0.001 | 1.01e-03 | 4 | ns_cubic15 (1) |  |
| parametric | ns_cubic5 | 1 | 0.001 | 1.01e-03 | 4 | ns_quintic5_f32 (1) |  |
| parametric | ns_cubic15 | 1 | 0.001 | 9.92e-04 | 5 | ns_quintic3 (1) |  |
| parametric | spec_p0.5 | 1 | 0.001 | 1.01e-03 | 7 | col_norm (1) | (+0.401) (+0.5) |
| parametric | dpre_-.25_-.25 | 1 | 0.001 | 9.96e-04 | 2 | cpolar_dshampoo (1) | (-0.250,-0.225) (-0.25,-0.25) |
| parametric | cpolar_rowwhite | 1 | 0.001 | 1.01e-03 | 1 | row_norm (1) | (-0.232,-0.268,+0.124,-0.085) (+0,-0.5,+0,+0) |
| parametric | cpolar_colwhite | 1 | 0.001 | 1.00e-03 | 11 | ns_quintic5_f32 (1) | (-0.458,+0.458,-0.103,+0.415) (+0,+0,-0.5,+0) |
| parametric | cpolar_dshampoo | 1 | 0.001 | 9.96e-04 | 2 | dpre_-.25_-.25 (1) | (-0.125,-0.125,-0.461,-0.079) (+0,-0.25,-0.25,+0) |
| parametric | ns_quintic5_f32 | 1 | 0.001 | 1.02e-03 | 1 | ns_quintic8 (1) |  |
| parametric | ns_quintic5_bf16 | 1 | 0.001 | 1.01e-03 | 1 | col_norm (17.4) |  |
| parametric | cpolar(.3,-.4,-.2,.1) | 1 | 0.001 | nan | None | polar (nan) | (-0.532,+0.432,-0.606,+0.417) (+0.3,-0.4,-0.2,+0.1) |
| parametric | frob | 1 | 0.01 | 1.02e-02 | 3 | ns_quintic5_f32 (1) | (+0.400) (+1) |
| parametric | sign | 1 | 0.01 | 1.00e-02 | 1 | cpolar_rowwhite (1) |  |
| parametric | row_norm | 1 | 0.01 | 1.00e-02 | 3 | cpolar_rowwhite (1) | (-0.500,-0.105) (-0.5,+0) |
| parametric | col_norm | 1 | 0.01 | 1.00e-02 | 3 | ns_quintic5_f32 (1) | (-0.000,-0.915) (+0,-0.5) |
| parametric | polar | 1 | 0.01 | 1.01e-02 | 5 | col_norm (1) | (+0.385) (+0) |
| parametric | ns_quintic5 | 1 | 0.01 | 1.01e-02 | 5 | col_norm (1) | (+3.310,-5.063,+1.252) (+3.44,-4.78,+2.03) |
| parametric | ns_quintic3 | 1 | 0.01 | 1.01e-02 | 4 | ns_quintic5_f32 (1) |  |
| parametric | ns_quintic8 | 1 | 0.01 | 1.01e-02 | 4 | ns_cubic15 (1) |  |
| parametric | ns_cubic5 | 1 | 0.01 | 1.01e-02 | 6 | ns_quintic5_f32 (1) |  |
| parametric | ns_cubic15 | 1 | 0.01 | 9.92e-03 | 8 | col_norm (1) |  |
| parametric | spec_p0.5 | 1 | 0.01 | 1.01e-02 | 9 | col_norm (1) | (+0.423) (+0.5) |
| parametric | dpre_-.25_-.25 | 1 | 0.01 | 9.96e-03 | 1 | cpolar_dshampoo (1) | (-0.250,-0.165) (-0.25,-0.25) |
| parametric | cpolar_rowwhite | 1 | 0.01 | 1.01e-02 | 1 | row_norm (1) | (-0.232,-0.268,+0.355,+0.124) (+0,-0.5,+0,+0) |
| parametric | cpolar_colwhite | 1 | 0.01 | 1.00e-02 | 11 | ns_quintic5_f32 (1) | (-0.458,+0.458,-0.621,-0.622) (+0,+0,-0.5,+0) |
| parametric | cpolar_dshampoo | 1 | 0.01 | 9.96e-03 | 2 | dpre_-.25_-.25 (1) | (-0.125,-0.125,-0.459,-0.080) (+0,-0.25,-0.25,+0) |
| parametric | ns_quintic5_f32 | 1 | 0.01 | 1.02e-02 | 1 | col_norm (1) |  |
| parametric | ns_quintic5_bf16 | 1 | 0.01 | 1.01e-02 | 1 | col_norm (2) |  |
| parametric | cpolar(.3,-.4,-.2,.1) | 1 | 0.01 | nan | None | polar (nan) | (-0.531,+0.431,-0.626,-0.101) (+0.3,-0.4,-0.2,+0.1) |
| parametric | frob | 4 | 1e-06 | 9.95e-07 | 1 | spec_p0.5 (2.58e+05) | (+1.000) (+1) |
| parametric | sign | 4 | 1e-06 | 1.00e-06 | 1 | row_norm (5.98e+05) |  |
| parametric | row_norm | 4 | 1e-06 | 1.00e-06 | 1 | sign (5.59e+05) | (-0.500,+0.000) (-0.5,+0) |
| parametric | col_norm | 4 | 1e-06 | 9.95e-07 | 1 | spec_p0.5 (4.62e+05) | (-0.000,-0.500) (+0,-0.5) |
| parametric | polar | 4 | 1e-06 | 1.01e-06 | 2 | ns_cubic15 (1) | (-0.000) (+0) |
| parametric | ns_quintic5 | 4 | 1e-06 | 1.00e-06 | 1 | ns_quintic5_f32 (1.3) | (+3.445,-4.775,+2.032) (+3.44,-4.78,+2.03) |
| parametric | ns_quintic3 | 4 | 1e-06 | 9.97e-07 | 1 | ns_quintic5_f32 (1.97e+05) |  |
| parametric | ns_quintic8 | 4 | 1e-06 | 1.00e-06 | 1 | ns_cubic15 (1.68e+05) |  |
| parametric | ns_cubic5 | 4 | 1e-06 | 9.95e-07 | 1 | spec_p0.5 (2.63e+05) |  |
| parametric | ns_cubic15 | 4 | 1e-06 | 1.00e-06 | 1 | polar (1) |  |
| parametric | spec_p0.5 | 4 | 1e-06 | 1.00e-06 | 1 | frob (2.78e+05) | (+0.500) (+0.5) |
| parametric | dpre_-.25_-.25 | 4 | 1e-06 | 1.00e-06 | 1 | spec_p0.5 (5.41e+05) | (-0.250,-0.250) (-0.25,-0.25) |
| parametric | cpolar_rowwhite | 4 | 1e-06 | 9.97e-07 | 1 | cpolar_dshampoo (5.55e+05) | (-0.000,-0.500,+0.000,-0.000) (+0,-0.5,+0,+0) |
| parametric | cpolar_colwhite | 4 | 1e-06 | 9.95e-07 | 1 | polar (3.65e+05) | (+0.000,+0.000,-0.500,-0.000) (+0,+0,-0.5,+0) |
| parametric | cpolar_dshampoo | 4 | 1e-06 | 1.00e-06 | 1 | ns_cubic15 (5.4e+05) | (+0.000,-0.250,-0.250,-0.000) (+0,-0.25,-0.25,+0) |
| parametric | ns_quintic5_f32 | 4 | 1e-06 | 1.01e-06 | 1 | ns_quintic5 (1.15) |  |
| parametric | ns_quintic5_bf16 | 4 | 1e-06 | 1.00e-06 | 1 | ns_quintic5 (3.67e+04) |  |
| parametric | cpolar(.3,-.4,-.2,.1) | 4 | 1e-06 | nan | None | spec_p0.5 (nan) | (+0.300,-0.400,-0.200,+0.100) (+0.3,-0.4,-0.2,+0.1) |
| parametric | frob | 4 | 0.001 | 9.95e-04 | 1 | spec_p0.5 (258) | (+1.000) (+1) |
| parametric | sign | 4 | 0.001 | 1.00e-03 | 1 | row_norm (598) |  |
| parametric | row_norm | 4 | 0.001 | 1.00e-03 | 1 | sign (559) | (-0.500,+0.000) (-0.5,+0) |
| parametric | col_norm | 4 | 0.001 | 9.95e-04 | 1 | spec_p0.5 (462) | (-0.000,-0.500) (+0,-0.5) |
| parametric | polar | 4 | 0.001 | 1.01e-03 | 2 | ns_cubic15 (1) | (-0.000) (+0) |
| parametric | ns_quintic5 | 4 | 0.001 | 1.00e-03 | 2 | ns_quintic5_f32 (1) | (+3.445,-4.775,+2.031) (+3.44,-4.78,+2.03) |
| parametric | ns_quintic3 | 4 | 0.001 | 9.97e-04 | 1 | ns_quintic5_f32 (197) |  |
| parametric | ns_quintic8 | 4 | 0.001 | 1.00e-03 | 1 | ns_cubic15 (168) |  |
| parametric | ns_cubic5 | 4 | 0.001 | 9.95e-04 | 1 | spec_p0.5 (263) |  |
| parametric | ns_cubic15 | 4 | 0.001 | 1.00e-03 | 1 | polar (1) |  |
| parametric | spec_p0.5 | 4 | 0.001 | 1.00e-03 | 1 | frob (278) | (+0.500) (+0.5) |
| parametric | dpre_-.25_-.25 | 4 | 0.001 | 1.00e-03 | 1 | spec_p0.5 (541) | (-0.250,-0.250) (-0.25,-0.25) |
| parametric | cpolar_rowwhite | 4 | 0.001 | 9.97e-04 | 1 | cpolar_dshampoo (555) | (-0.000,-0.500,+0.000,-0.000) (+0,-0.5,+0,+0) |
| parametric | cpolar_colwhite | 4 | 0.001 | 9.95e-04 | 1 | polar (365) | (+0.000,+0.000,-0.500,-0.000) (+0,+0,-0.5,+0) |
| parametric | cpolar_dshampoo | 4 | 0.001 | 1.00e-03 | 1 | ns_cubic15 (540) | (+0.000,-0.250,-0.250,-0.000) (+0,-0.25,-0.25,+0) |
| parametric | ns_quintic5_f32 | 4 | 0.001 | 1.01e-03 | 1 | ns_quintic5 (1) |  |
| parametric | ns_quintic5_bf16 | 4 | 0.001 | 1.00e-03 | 1 | ns_quintic5 (36.7) |  |
| parametric | cpolar(.3,-.4,-.2,.1) | 4 | 0.001 | nan | None | spec_p0.5 (nan) | (+0.300,-0.400,-0.200,+0.100) (+0.3,-0.4,-0.2,+0.1) |
| parametric | frob | 4 | 0.01 | 9.95e-03 | 1 | spec_p0.5 (25.9) | (+1.000) (+1) |
| parametric | sign | 4 | 0.01 | 1.00e-02 | 1 | row_norm (59.8) |  |
| parametric | row_norm | 4 | 0.01 | 1.00e-02 | 1 | sign (55.9) | (-0.500,+0.000) (-0.5,+0) |
| parametric | col_norm | 4 | 0.01 | 9.95e-03 | 1 | spec_p0.5 (46.2) | (-0.000,-0.500) (+0,-0.5) |
| parametric | polar | 4 | 0.01 | 1.01e-02 | 2 | ns_cubic15 (1) | (-0.000) (+0) |
| parametric | ns_quintic5 | 4 | 0.01 | 1.00e-02 | 2 | ns_quintic5_f32 (1) | (+3.445,-4.775,+2.031) (+3.44,-4.78,+2.03) |
| parametric | ns_quintic3 | 4 | 0.01 | 9.97e-03 | 1 | ns_quintic5_f32 (19.7) |  |
| parametric | ns_quintic8 | 4 | 0.01 | 1.00e-02 | 1 | ns_cubic15 (16.8) |  |
| parametric | ns_cubic5 | 4 | 0.01 | 9.95e-03 | 1 | spec_p0.5 (26.3) |  |
| parametric | ns_cubic15 | 4 | 0.01 | 1.00e-02 | 1 | polar (1) |  |
| parametric | spec_p0.5 | 4 | 0.01 | 1.00e-02 | 1 | frob (27.8) | (+0.500) (+0.5) |
| parametric | dpre_-.25_-.25 | 4 | 0.01 | 1.00e-02 | 1 | spec_p0.5 (54.1) | (-0.250,-0.250) (-0.25,-0.25) |
| parametric | cpolar_rowwhite | 4 | 0.01 | 9.97e-03 | 1 | cpolar_dshampoo (55.5) | (-0.000,-0.500,+0.000,-0.000) (+0,-0.5,+0,+0) |
| parametric | cpolar_colwhite | 4 | 0.01 | 9.95e-03 | 1 | polar (36.5) | (+0.000,+0.000,-0.500,-0.000) (+0,+0,-0.5,+0) |
| parametric | cpolar_dshampoo | 4 | 0.01 | 1.00e-02 | 1 | ns_cubic15 (54) | (+0.000,-0.250,-0.250,-0.000) (+0,-0.25,-0.25,+0) |
| parametric | ns_quintic5_f32 | 4 | 0.01 | 1.01e-02 | 1 | ns_quintic5 (1) |  |
| parametric | ns_quintic5_bf16 | 4 | 0.01 | 1.00e-02 | 1 | ns_quintic5 (3.79) |  |
| parametric | cpolar(.3,-.4,-.2,.1) | 4 | 0.01 | nan | None | spec_p0.5 (nan) | (+0.300,-0.400,-0.200,+0.100) (+0.3,-0.4,-0.2,+0.1) |
| parametric | frob | 16 | 1e-06 | 1.00e-06 | 1 | spec_p0.5 (3.24e+05) | (+1.000) (+1) |
| parametric | sign | 16 | 1e-06 | 9.99e-07 | 1 | row_norm (6.79e+05) |  |
| parametric | row_norm | 16 | 1e-06 | 9.99e-07 | 1 | dpre_-.25_-.25 (5.25e+05) | (-0.500,+0.000) (-0.5,+0) |
| parametric | col_norm | 16 | 1e-06 | 1.00e-06 | 1 | spec_p0.5 (4.43e+05) | (-0.000,-0.500) (+0,-0.5) |
| parametric | polar | 16 | 1e-06 | 1.00e-06 | 1 | ns_cubic15 (1) | (-0.000) (+0) |
| parametric | ns_quintic5 | 16 | 1e-06 | 9.97e-07 | 1 | ns_quintic5_f32 (1.33) | (+3.444,-4.775,+2.032) (+3.44,-4.78,+2.03) |
| parametric | ns_quintic3 | 16 | 1e-06 | 9.98e-07 | 1 | ns_quintic5_f32 (2.31e+05) |  |
| parametric | ns_quintic8 | 16 | 1e-06 | 1.00e-06 | 1 | ns_cubic15 (2.05e+05) |  |
| parametric | ns_cubic5 | 16 | 1e-06 | 1.00e-06 | 1 | spec_p0.5 (1.86e+05) |  |
| parametric | ns_cubic15 | 16 | 1e-06 | 9.99e-07 | 1 | polar (4.92e+03) |  |
| parametric | spec_p0.5 | 16 | 1e-06 | 1.00e-06 | 1 | ns_cubic5 (2.04e+05) | (+0.500) (+0.5) |
| parametric | dpre_-.25_-.25 | 16 | 1e-06 | 9.97e-07 | 1 | col_norm (4.99e+05) | (-0.250,-0.250) (-0.25,-0.25) |
| parametric | cpolar_rowwhite | 16 | 1e-06 | 9.99e-07 | 1 | cpolar_dshampoo (4.98e+05) | (+0.000,-0.500,-0.000,+0.000) (+0,-0.5,+0,+0) |
| parametric | cpolar_colwhite | 16 | 1e-06 | 9.99e-07 | 1 | ns_cubic15 (2.63e+05) | (-0.000,+0.000,-0.500,+0.000) (+0,+0,-0.5,+0) |
| parametric | cpolar_dshampoo | 16 | 1e-06 | 1.00e-06 | 1 | polar (4.68e+05) | (-0.000,-0.250,-0.250,-0.000) (+0,-0.25,-0.25,+0) |
| parametric | ns_quintic5_f32 | 16 | 1e-06 | 9.99e-07 | 1 | ns_quintic5 (1.46) |  |
| parametric | ns_quintic5_bf16 | 16 | 1e-06 | 9.99e-07 | 1 | ns_quintic5 (3.28e+04) |  |
| parametric | cpolar(.3,-.4,-.2,.1) | 16 | 1e-06 | nan | None | ns_cubic15 (nan) | (+0.300,-0.400,-0.200,+0.100) (+0.3,-0.4,-0.2,+0.1) |
| parametric | frob | 16 | 0.001 | 1.00e-03 | 1 | spec_p0.5 (324) | (+1.000) (+1) |
| parametric | sign | 16 | 0.001 | 9.99e-04 | 1 | row_norm (679) |  |
| parametric | row_norm | 16 | 0.001 | 9.99e-04 | 1 | dpre_-.25_-.25 (525) | (-0.500,+0.000) (-0.5,+0) |
| parametric | col_norm | 16 | 0.001 | 1.00e-03 | 1 | spec_p0.5 (443) | (-0.000,-0.500) (+0,-0.5) |
| parametric | polar | 16 | 0.001 | 1.00e-03 | 1 | ns_cubic15 (1) | (+0.000) (+0) |
| parametric | ns_quintic5 | 16 | 0.001 | 9.97e-04 | 2 | ns_quintic5_f32 (1) | (+3.444,-4.775,+2.032) (+3.44,-4.78,+2.03) |
| parametric | ns_quintic3 | 16 | 0.001 | 9.98e-04 | 1 | ns_quintic5_f32 (231) |  |
| parametric | ns_quintic8 | 16 | 0.001 | 1.00e-03 | 1 | ns_cubic15 (205) |  |
| parametric | ns_cubic5 | 16 | 0.001 | 1.00e-03 | 1 | spec_p0.5 (186) |  |
| parametric | ns_cubic15 | 16 | 0.001 | 9.99e-04 | 1 | polar (5.02) |  |
| parametric | spec_p0.5 | 16 | 0.001 | 1.00e-03 | 1 | ns_cubic5 (204) | (+0.500) (+0.5) |
| parametric | dpre_-.25_-.25 | 16 | 0.001 | 9.97e-04 | 1 | col_norm (499) | (-0.250,-0.250) (-0.25,-0.25) |
| parametric | cpolar_rowwhite | 16 | 0.001 | 9.99e-04 | 1 | cpolar_dshampoo (498) | (+0.000,-0.500,-0.000,+0.000) (+0,-0.5,+0,+0) |
| parametric | cpolar_colwhite | 16 | 0.001 | 9.99e-04 | 1 | ns_cubic15 (263) | (-0.000,+0.000,-0.500,+0.000) (+0,+0,-0.5,+0) |
| parametric | cpolar_dshampoo | 16 | 0.001 | 1.00e-03 | 1 | polar (468) | (-0.000,-0.250,-0.250,-0.000) (+0,-0.25,-0.25,+0) |
| parametric | ns_quintic5_f32 | 16 | 0.001 | 9.99e-04 | 1 | ns_quintic5 (1) |  |
| parametric | ns_quintic5_bf16 | 16 | 0.001 | 9.99e-04 | 1 | ns_quintic5 (32.8) |  |
| parametric | cpolar(.3,-.4,-.2,.1) | 16 | 0.001 | nan | None | ns_cubic15 (nan) | (+0.300,-0.400,-0.200,+0.100) (+0.3,-0.4,-0.2,+0.1) |
| parametric | frob | 16 | 0.01 | 1.00e-02 | 1 | spec_p0.5 (32.4) | (+1.000) (+1) |
| parametric | sign | 16 | 0.01 | 9.99e-03 | 1 | row_norm (67.9) |  |
| parametric | row_norm | 16 | 0.01 | 9.99e-03 | 1 | dpre_-.25_-.25 (52.5) | (-0.500,+0.000) (-0.5,+0) |
| parametric | col_norm | 16 | 0.01 | 1.00e-02 | 1 | spec_p0.5 (44.3) | (-0.000,-0.500) (+0,-0.5) |
| parametric | polar | 16 | 0.01 | 1.00e-02 | 1 | ns_cubic15 (1) | (+0.000) (+0) |
| parametric | ns_quintic5 | 16 | 0.01 | 9.97e-03 | 2 | ns_quintic5_f32 (1) | (+3.444,-4.775,+2.032) (+3.44,-4.78,+2.03) |
| parametric | ns_quintic3 | 16 | 0.01 | 9.98e-03 | 1 | ns_quintic5_f32 (23.1) |  |
| parametric | ns_quintic8 | 16 | 0.01 | 1.00e-02 | 1 | ns_cubic15 (20.5) |  |
| parametric | ns_cubic5 | 16 | 0.01 | 1.00e-02 | 1 | spec_p0.5 (18.6) |  |
| parametric | ns_cubic15 | 16 | 0.01 | 9.99e-03 | 1 | polar (1.11) |  |
| parametric | spec_p0.5 | 16 | 0.01 | 1.00e-02 | 1 | ns_cubic5 (20.4) | (+0.500) (+0.5) |
| parametric | dpre_-.25_-.25 | 16 | 0.01 | 9.97e-03 | 1 | col_norm (50) | (-0.250,-0.250) (-0.25,-0.25) |
| parametric | cpolar_rowwhite | 16 | 0.01 | 9.99e-03 | 1 | cpolar_dshampoo (49.8) | (+0.000,-0.500,-0.000,+0.000) (+0,-0.5,+0,+0) |
| parametric | cpolar_colwhite | 16 | 0.01 | 9.99e-03 | 1 | ns_cubic15 (26.3) | (-0.000,+0.000,-0.500,+0.000) (+0,+0,-0.5,+0) |
| parametric | cpolar_dshampoo | 16 | 0.01 | 1.00e-02 | 1 | polar (46.8) | (-0.000,-0.250,-0.250,-0.000) (+0,-0.25,-0.25,+0) |
| parametric | ns_quintic5_f32 | 16 | 0.01 | 9.99e-03 | 1 | ns_quintic5 (1) |  |
| parametric | ns_quintic5_bf16 | 16 | 0.01 | 9.99e-03 | 1 | ns_quintic5 (3.42) |  |
| parametric | cpolar(.3,-.4,-.2,.1) | 16 | 0.01 | nan | None | ns_cubic15 (nan) | (+0.300,-0.400,-0.200,+0.100) (+0.3,-0.4,-0.2,+0.1) |
| parametric | frob | 64 | 1e-06 | 1.00e-06 | 1 | ns_cubic5 (3.4e+05) | (+1.000) (+1) |
| parametric | sign | 64 | 1e-06 | 1.00e-06 | 1 | row_norm (6.94e+05) |  |
| parametric | row_norm | 64 | 1e-06 | 1.00e-06 | 1 | dpre_-.25_-.25 (5.06e+05) | (-0.500,-0.000) (-0.5,+0) |
| parametric | col_norm | 64 | 1e-06 | 1.00e-06 | 1 | ns_cubic5 (4.41e+05) | (+0.000,-0.500) (+0,-0.5) |
| parametric | polar | 64 | 1e-06 | 9.98e-07 | 1 | ns_cubic15 (3.46e+04) | (+0.000) (+0) |
| parametric | ns_quintic5 | 64 | 1e-06 | 9.99e-07 | 1 | ns_quintic5_f32 (1.55) | (+3.444,-4.775,+2.032) (+3.44,-4.78,+2.03) |
| parametric | ns_quintic3 | 64 | 1e-06 | 1.00e-06 | 1 | ns_cubic15 (3.57e+05) |  |
| parametric | ns_quintic8 | 64 | 1e-06 | 9.99e-07 | 1 | ns_cubic15 (1.99e+05) |  |
| parametric | ns_cubic5 | 64 | 1e-06 | 9.99e-07 | 1 | spec_p0.5 (1.8e+05) |  |
| parametric | ns_cubic15 | 64 | 1e-06 | 9.99e-07 | 1 | polar (1.53e+04) |  |
| parametric | spec_p0.5 | 64 | 1e-06 | 9.98e-07 | 1 | ns_cubic5 (1.77e+05) | (+0.500) (+0.5) |
| parametric | dpre_-.25_-.25 | 64 | 1e-06 | 9.99e-07 | 1 | ns_cubic5 (4.63e+05) | (-0.250,-0.250) (-0.25,-0.25) |
| parametric | cpolar_rowwhite | 64 | 1e-06 | 9.99e-07 | 1 | cpolar_dshampoo (4.61e+05) | (+0.000,-0.500,+0.000,-0.000) (+0,-0.5,+0,+0) |
| parametric | cpolar_colwhite | 64 | 1e-06 | 9.98e-07 | 1 | polar (2.72e+05) | (-0.000,+0.000,-0.500,+0.000) (+0,+0,-0.5,+0) |
| parametric | cpolar_dshampoo | 64 | 1e-06 | 1.00e-06 | 1 | polar (4.23e+05) | (+0.000,-0.250,-0.250,+0.000) (+0,-0.25,-0.25,+0) |
| parametric | ns_quintic5_f32 | 64 | 1e-06 | 9.99e-07 | 1 | ns_quintic5 (1.54) |  |
| parametric | ns_quintic5_bf16 | 64 | 1e-06 | 9.99e-07 | 1 | ns_quintic5 (3.85e+04) |  |
| parametric | cpolar(.3,-.4,-.2,.1) | 64 | 1e-06 | nan | None | ns_cubic15 (nan) | (+0.300,-0.400,-0.200,+0.100) (+0.3,-0.4,-0.2,+0.1) |
| parametric | frob | 64 | 0.001 | 1.00e-03 | 1 | ns_cubic5 (340) | (+1.000) (+1) |
| parametric | sign | 64 | 0.001 | 1.00e-03 | 1 | row_norm (694) |  |
| parametric | row_norm | 64 | 0.001 | 1.00e-03 | 1 | dpre_-.25_-.25 (506) | (-0.500,-0.000) (-0.5,+0) |
| parametric | col_norm | 64 | 0.001 | 1.00e-03 | 1 | ns_cubic5 (441) | (+0.000,-0.500) (+0,-0.5) |
| parametric | polar | 64 | 0.001 | 9.98e-04 | 1 | ns_cubic15 (34.6) | (+0.000) (+0) |
| parametric | ns_quintic5 | 64 | 0.001 | 9.99e-04 | 1 | ns_quintic5_f32 (1) | (+3.444,-4.775,+2.032) (+3.44,-4.78,+2.03) |
| parametric | ns_quintic3 | 64 | 0.001 | 1.00e-03 | 1 | ns_cubic15 (357) |  |
| parametric | ns_quintic8 | 64 | 0.001 | 9.99e-04 | 1 | ns_cubic15 (199) |  |
| parametric | ns_cubic5 | 64 | 0.001 | 9.99e-04 | 1 | spec_p0.5 (180) |  |
| parametric | ns_cubic15 | 64 | 0.001 | 9.99e-04 | 1 | polar (15.4) |  |
| parametric | spec_p0.5 | 64 | 0.001 | 9.98e-04 | 1 | ns_cubic5 (177) | (+0.500) (+0.5) |
| parametric | dpre_-.25_-.25 | 64 | 0.001 | 9.99e-04 | 1 | ns_cubic5 (463) | (-0.250,-0.250) (-0.25,-0.25) |
| parametric | cpolar_rowwhite | 64 | 0.001 | 9.99e-04 | 1 | cpolar_dshampoo (461) | (+0.000,-0.500,+0.000,-0.000) (+0,-0.5,+0,+0) |
| parametric | cpolar_colwhite | 64 | 0.001 | 9.98e-04 | 1 | polar (272) | (-0.000,+0.000,-0.500,+0.000) (+0,+0,-0.5,+0) |
| parametric | cpolar_dshampoo | 64 | 0.001 | 1.00e-03 | 1 | polar (423) | (+0.000,-0.250,-0.250,+0.000) (+0,-0.25,-0.25,+0) |
| parametric | ns_quintic5_f32 | 64 | 0.001 | 9.99e-04 | 1 | ns_quintic5 (1) |  |
| parametric | ns_quintic5_bf16 | 64 | 0.001 | 9.99e-04 | 1 | ns_quintic5 (38.5) |  |
| parametric | cpolar(.3,-.4,-.2,.1) | 64 | 0.001 | nan | None | ns_cubic15 (nan) | (+0.300,-0.400,-0.200,+0.100) (+0.3,-0.4,-0.2,+0.1) |
| parametric | frob | 64 | 0.01 | 1.00e-02 | 1 | ns_cubic5 (34) | (+1.000) (+1) |
| parametric | sign | 64 | 0.01 | 1.00e-02 | 1 | row_norm (69.4) |  |
| parametric | row_norm | 64 | 0.01 | 1.00e-02 | 1 | dpre_-.25_-.25 (50.6) | (-0.500,-0.000) (-0.5,+0) |
| parametric | col_norm | 64 | 0.01 | 1.00e-02 | 1 | ns_cubic5 (44.2) | (+0.000,-0.500) (+0,-0.5) |
| parametric | polar | 64 | 0.01 | 9.98e-03 | 1 | ns_cubic15 (3.6) | (+0.000) (+0) |
| parametric | ns_quintic5 | 64 | 0.01 | 9.99e-03 | 1 | ns_quintic5_f32 (1) | (+3.444,-4.775,+2.032) (+3.44,-4.78,+2.03) |
| parametric | ns_quintic3 | 64 | 0.01 | 1.00e-02 | 1 | ns_cubic15 (35.7) |  |
| parametric | ns_quintic8 | 64 | 0.01 | 9.99e-03 | 1 | ns_cubic15 (20) |  |
| parametric | ns_cubic5 | 64 | 0.01 | 9.99e-03 | 1 | spec_p0.5 (18) |  |
| parametric | ns_cubic15 | 64 | 0.01 | 9.99e-03 | 1 | polar (1.83) |  |
| parametric | spec_p0.5 | 64 | 0.01 | 9.98e-03 | 1 | ns_cubic5 (17.7) | (+0.500) (+0.5) |
| parametric | dpre_-.25_-.25 | 64 | 0.01 | 9.99e-03 | 1 | ns_cubic5 (46.3) | (-0.250,-0.250) (-0.25,-0.25) |
| parametric | cpolar_rowwhite | 64 | 0.01 | 9.99e-03 | 1 | cpolar_dshampoo (46.1) | (+0.000,-0.500,+0.000,-0.000) (+0,-0.5,+0,+0) |
| parametric | cpolar_colwhite | 64 | 0.01 | 9.98e-03 | 1 | polar (27.2) | (-0.000,+0.000,-0.500,+0.000) (+0,+0,-0.5,+0) |
| parametric | cpolar_dshampoo | 64 | 0.01 | 1.00e-02 | 1 | polar (42.3) | (+0.000,-0.250,-0.250,+0.000) (+0,-0.25,-0.25,+0) |
| parametric | ns_quintic5_f32 | 64 | 0.01 | 9.99e-03 | 1 | ns_quintic5 (1) |  |
| parametric | ns_quintic5_bf16 | 64 | 0.01 | 9.99e-03 | 1 | ns_quintic5 (3.97) |  |
| parametric | cpolar(.3,-.4,-.2,.1) | 64 | 0.01 | nan | None | ns_cubic15 (nan) | (+0.300,-0.400,-0.200,+0.100) (+0.3,-0.4,-0.2,+0.1) |
| sketch | frob | 1 | 1e-06 | 1.02e-06 | 9 | ns_cubic15 (1) | (+1.044) (+1) |
| sketch | sign | 1 | 1e-06 | 9.91e-07 | 1 | row_norm (1) |  |
| sketch | row_norm | 1 | 1e-06 | 9.98e-07 | 3 | cpolar_rowwhite (1) | (-0.500,-0.104) (-0.5,+0) |
| sketch | col_norm | 1 | 1e-06 | 9.91e-07 | 10 | ns_quintic5 (1) | (-0.000,+0.620) (+0,-0.5) |
| sketch | polar | 1 | 1e-06 | 1.01e-06 | 8 | col_norm (1) | (-0.313) (+0) |
| sketch | ns_quintic5 | 1 | 1e-06 | 9.91e-07 | 8 | ns_quintic3 (1) | (+2.761,-2.146,+0.914) (+3.44,-4.78,+2.03) |
| sketch | ns_quintic3 | 1 | 1e-06 | 1.01e-06 | 10 | cpolar_colwhite (1) |  |
| sketch | ns_quintic8 | 1 | 1e-06 | 1.00e-06 | 9 | apolar_-.5 (1) |  |
| sketch | ns_cubic5 | 1 | 1e-06 | 1.01e-06 | 6 | ns_quintic3 (1) |  |
| sketch | ns_cubic15 | 1 | 1e-06 | 9.95e-07 | 3 | ns_cubic5 (1) |  |
| sketch | spec_p0.5 | 1 | 1e-06 | 1.00e-06 | 10 | cpolar_colwhite (1) | (+1.128) (+0.5) |
| sketch | dpre_-.25_-.25 | 1 | 1e-06 | 9.86e-07 | 1 | cpolar_dshampoo (1) | (-0.250,+0.159) (-0.25,-0.25) |
| sketch | cpolar_rowwhite | 1 | 1e-06 | 9.88e-07 | 3 | row_norm (1) | (-0.754,+0.254,+0.386,+0.430) (+0,-0.5,+0,+0) |
| sketch | cpolar_colwhite | 1 | 1e-06 | 1.01e-06 | 10 | spec_p0.5 (1) | (-0.458,+0.458,-0.622,-0.107) (+0,+0,-0.5,+0) |
| sketch | cpolar_dshampoo | 1 | 1e-06 | 1.01e-06 | 1 | dpre_-.25_-.25 (1) | (-0.125,-0.125,-0.461,-0.079) (+0,-0.25,-0.25,+0) |
| sketch | ns_quintic5_f32 | 1 | 1e-06 | 9.95e-07 | 1 | ns_cubic15 (1.04) |  |
| sketch | ns_quintic5_bf16 | 1 | 1e-06 | 1.02e-06 | 1 | ns_quintic5_f32 (1.86e+04) |  |
| sketch | apolar_-.5 | 1 | 1e-06 | 1.02e-06 | 1 | col_norm (1) | (-0.613,-0.101) (-0.5,+0) |
| sketch | cpolar(.3,-.4,-.2,.1) | 1 | 1e-06 | nan | None | ns_quintic5_bf16 (nan) | (-0.531,+0.431,-0.626,-0.608) (+0.3,-0.4,-0.2,+0.1) |
| sketch | frob | 1 | 0.001 | 1.02e-03 | 10 | ns_quintic5_f32 (1) | (+1.045) (+1) |
| sketch | sign | 1 | 0.001 | 9.91e-04 | 1 | row_norm (1) |  |
| sketch | row_norm | 1 | 0.001 | 9.98e-04 | 3 | cpolar_rowwhite (1) | (-0.500,-0.620) (-0.5,+0) |
| sketch | col_norm | 1 | 0.001 | 9.91e-04 | 10 | ns_quintic5_f32 (1) | (-0.000,-0.187) (+0,-0.5) |
| sketch | polar | 1 | 0.001 | 1.01e-03 | 9 | ns_quintic5_f32 (1) | (-0.301) (+0) |
| sketch | ns_quintic5 | 1 | 0.001 | 9.91e-04 | 6 | ns_quintic5_f32 (1) | (+3.694,-4.866,+1.912) (+3.44,-4.78,+2.03) |
| sketch | ns_quintic3 | 1 | 0.001 | 1.01e-03 | 8 | ns_quintic5_f32 (1) |  |
| sketch | ns_quintic8 | 1 | 0.001 | 1.00e-03 | 10 | ns_quintic5_f32 (1) |  |
| sketch | ns_cubic5 | 1 | 0.001 | 1.01e-03 | 4 | col_norm (1) |  |
| sketch | ns_cubic15 | 1 | 0.001 | 9.95e-04 | 6 | ns_quintic8 (1) |  |
| sketch | spec_p0.5 | 1 | 0.001 | 1.00e-03 | 11 | ns_quintic5_f32 (1) | (+0.395) (+0.5) |
| sketch | dpre_-.25_-.25 | 1 | 0.001 | 9.86e-04 | 1 | cpolar_dshampoo (1) | (-0.250,+0.008) (-0.25,-0.25) |
| sketch | cpolar_rowwhite | 1 | 0.001 | 9.88e-04 | 3 | row_norm (1) | (-0.232,-0.268,+0.357,+0.119) (+0,-0.5,+0,+0) |
| sketch | cpolar_colwhite | 1 | 0.001 | 1.01e-03 | 11 | ns_quintic5_f32 (1) | (-0.458,+0.458,-0.620,-0.622) (+0,+0,-0.5,+0) |
| sketch | cpolar_dshampoo | 1 | 0.001 | 1.01e-03 | 1 | dpre_-.25_-.25 (1) | (-0.125,-0.125,-0.461,-0.477) (+0,-0.25,-0.25,+0) |
| sketch | ns_quintic5_f32 | 1 | 0.001 | 9.95e-04 | 12 | cpolar_colwhite (1) |  |
| sketch | ns_quintic5_bf16 | 1 | 0.001 | 1.02e-03 | 1 | ns_quintic5_f32 (18.6) |  |
| sketch | apolar_-.5 | 1 | 0.001 | 1.02e-03 | 9 | ns_quintic5_f32 (1) | (-0.630,-0.090) (-0.5,+0) |
| sketch | cpolar(.3,-.4,-.2,.1) | 1 | 0.001 | nan | None | ns_quintic5_bf16 (nan) | (-0.531,+0.431,-0.626,+0.406) (+0.3,-0.4,-0.2,+0.1) |
| sketch | frob | 1 | 0.01 | 1.02e-02 | 9 | ns_quintic5_f32 (1) | (-0.313) (+1) |
| sketch | sign | 1 | 0.01 | 9.91e-03 | 1 | row_norm (1) |  |
| sketch | row_norm | 1 | 0.01 | 9.98e-03 | 2 | cpolar_rowwhite (1) | (-0.500,-0.631) (-0.5,+0) |
| sketch | col_norm | 1 | 0.01 | 9.91e-03 | 11 | ns_quintic5_f32 (1) | (-0.000,-0.930) (+0,-0.5) |
| sketch | polar | 1 | 0.01 | 1.01e-02 | 4 | ns_quintic5_f32 (1) | (-0.317) (+0) |
| sketch | ns_quintic5 | 1 | 0.01 | 9.91e-03 | 6 | ns_quintic5_f32 (1) | (+3.596,-5.021,+0.493) (+3.44,-4.78,+2.03) |
| sketch | ns_quintic3 | 1 | 0.01 | 1.01e-02 | 11 | ns_quintic5_f32 (1) |  |
| sketch | ns_quintic8 | 1 | 0.01 | 1.00e-02 | 5 | ns_quintic5_f32 (1) |  |
| sketch | ns_cubic5 | 1 | 0.01 | 1.01e-02 | 5 | ns_quintic3 (1) |  |
| sketch | ns_cubic15 | 1 | 0.01 | 9.95e-03 | 4 | ns_cubic5 (1) |  |
| sketch | spec_p0.5 | 1 | 0.01 | 1.00e-02 | 8 | ns_quintic5_f32 (1) | (+1.100) (+0.5) |
| sketch | dpre_-.25_-.25 | 1 | 0.01 | 9.86e-03 | 1 | cpolar_dshampoo (1) | (-0.250,-0.150) (-0.25,-0.25) |
| sketch | cpolar_rowwhite | 1 | 0.01 | 9.88e-03 | 3 | row_norm (1) | (-0.754,+0.254,-0.673,-0.644) (+0,-0.5,+0,+0) |
| sketch | cpolar_colwhite | 1 | 0.01 | 1.01e-02 | 12 | ns_quintic5_f32 (1) | (-0.458,+0.458,-0.641,+0.415) (+0,+0,-0.5,+0) |
| sketch | cpolar_dshampoo | 1 | 0.01 | 1.01e-02 | 1 | dpre_-.25_-.25 (1) | (-0.125,-0.125,-0.461,+0.318) (+0,-0.25,-0.25,+0) |
| sketch | ns_quintic5_f32 | 1 | 0.01 | 9.95e-03 | 12 | cpolar_colwhite (1) |  |
| sketch | ns_quintic5_bf16 | 1 | 0.01 | 1.02e-02 | 1 | ns_quintic5_f32 (2.12) |  |
| sketch | apolar_-.5 | 1 | 0.01 | 1.02e-02 | 2 | ns_quintic5_f32 (1) | (-0.600,+0.400) (-0.5,+0) |
| sketch | cpolar(.3,-.4,-.2,.1) | 1 | 0.01 | nan | None | ns_quintic5_bf16 (nan) | (-0.531,+0.431,-0.626,-0.609) (+0.3,-0.4,-0.2,+0.1) |
| sketch | frob | 4 | 1e-06 | 9.96e-07 | 1 | spec_p0.5 (2.76e+05) | (+1.000) (+1) |
| sketch | sign | 4 | 1e-06 | 1.01e-06 | 1 | row_norm (5.06e+05) |  |
| sketch | row_norm | 4 | 1e-06 | 1.00e-06 | 1 | cpolar_rowwhite (4.21e+05) | (-0.500,+0.000) (-0.5,+0) |
| sketch | col_norm | 4 | 1e-06 | 1.00e-06 | 1 | spec_p0.5 (2.62e+05) | (+0.000,-0.500) (+0,-0.5) |
| sketch | polar | 4 | 1e-06 | 1.00e-06 | 1 | ns_cubic15 (1) | (-0.000) (+0) |
| sketch | ns_quintic5 | 4 | 1e-06 | 9.99e-07 | 1 | ns_quintic5_f32 (1.32) | (+3.445,-4.775,+2.031) (+3.44,-4.78,+2.03) |
| sketch | ns_quintic3 | 4 | 1e-06 | 1.00e-06 | 1 | ns_quintic5 (5.56e+04) |  |
| sketch | ns_quintic8 | 4 | 1e-06 | 9.96e-07 | 1 | ns_cubic15 (1.69e+05) |  |
| sketch | ns_cubic5 | 4 | 1e-06 | 9.97e-07 | 1 | ns_cubic15 (1.03e+05) |  |
| sketch | ns_cubic15 | 4 | 1e-06 | 9.98e-07 | 2 | polar (1) |  |
| sketch | spec_p0.5 | 4 | 1e-06 | 1.00e-06 | 1 | frob (2.21e+05) | (+0.500) (+0.5) |
| sketch | dpre_-.25_-.25 | 4 | 1e-06 | 1.00e-06 | 1 | cpolar_dshampoo (3.82e+05) | (-0.250,-0.250) (-0.25,-0.25) |
| sketch | cpolar_rowwhite | 4 | 1e-06 | 1.00e-06 | 1 | row_norm (3.63e+05) | (+0.000,-0.500,+0.000,+0.000) (+0,-0.5,+0,+0) |
| sketch | cpolar_colwhite | 4 | 1e-06 | 1.01e-06 | 1 | ns_cubic15 (9.33e+04) | (-0.000,+0.000,-0.500,-0.000) (+0,+0,-0.5,+0) |
| sketch | cpolar_dshampoo | 4 | 1e-06 | 9.97e-07 | 1 | dpre_-.25_-.25 (4.4e+05) | (-0.000,-0.250,-0.250,-0.000) (+0,-0.25,-0.25,+0) |
| sketch | ns_quintic5_f32 | 4 | 1e-06 | 9.92e-07 | 1 | ns_quintic5 (1.22) |  |
| sketch | ns_quintic5_bf16 | 4 | 1e-06 | 1.01e-06 | 1 | ns_quintic5 (3.02e+04) |  |
| sketch | apolar_-.5 | 4 | 1e-06 | 1.00e-06 | 1 | ns_cubic15 (1.03e+04) | (-0.500,+0.000) (-0.5,+0) |
| sketch | cpolar(.3,-.4,-.2,.1) | 4 | 1e-06 | nan | None | ns_cubic5 (nan) | (+0.300,-0.400,-0.200,+0.100) (+0.3,-0.4,-0.2,+0.1) |
| sketch | frob | 4 | 0.001 | 9.96e-04 | 1 | spec_p0.5 (276) | (+1.000) (+1) |
| sketch | sign | 4 | 0.001 | 1.01e-03 | 1 | row_norm (506) |  |
| sketch | row_norm | 4 | 0.001 | 1.00e-03 | 1 | cpolar_rowwhite (421) | (-0.500,+0.000) (-0.5,+0) |
| sketch | col_norm | 4 | 0.001 | 1.00e-03 | 1 | spec_p0.5 (262) | (+0.000,-0.500) (+0,-0.5) |
| sketch | polar | 4 | 0.001 | 1.00e-03 | 1 | ns_cubic15 (1) | (-0.000) (+0) |
| sketch | ns_quintic5 | 4 | 0.001 | 9.99e-04 | 2 | ns_quintic5_f32 (1) | (+3.445,-4.775,+2.031) (+3.44,-4.78,+2.03) |
| sketch | ns_quintic3 | 4 | 0.001 | 1.00e-03 | 1 | ns_quintic5 (55.6) |  |
| sketch | ns_quintic8 | 4 | 0.001 | 9.96e-04 | 1 | ns_cubic15 (169) |  |
| sketch | ns_cubic5 | 4 | 0.001 | 9.97e-04 | 1 | ns_cubic15 (103) |  |
| sketch | ns_cubic15 | 4 | 0.001 | 9.98e-04 | 2 | polar (1) |  |
| sketch | spec_p0.5 | 4 | 0.001 | 1.00e-03 | 1 | frob (221) | (+0.500) (+0.5) |
| sketch | dpre_-.25_-.25 | 4 | 0.001 | 1.00e-03 | 1 | cpolar_dshampoo (382) | (-0.250,-0.250) (-0.25,-0.25) |
| sketch | cpolar_rowwhite | 4 | 0.001 | 1.00e-03 | 1 | row_norm (363) | (+0.000,-0.500,+0.000,+0.000) (+0,-0.5,+0,+0) |
| sketch | cpolar_colwhite | 4 | 0.001 | 1.01e-03 | 1 | ns_cubic15 (93.3) | (-0.000,+0.000,-0.500,-0.000) (+0,+0,-0.5,+0) |
| sketch | cpolar_dshampoo | 4 | 0.001 | 9.97e-04 | 1 | dpre_-.25_-.25 (440) | (-0.000,-0.250,-0.250,-0.000) (+0,-0.25,-0.25,+0) |
| sketch | ns_quintic5_f32 | 4 | 0.001 | 9.92e-04 | 2 | ns_quintic5 (1) |  |
| sketch | ns_quintic5_bf16 | 4 | 0.001 | 1.01e-03 | 1 | ns_quintic5 (30.2) |  |
| sketch | apolar_-.5 | 4 | 0.001 | 1.00e-03 | 1 | ns_cubic15 (10.3) | (-0.500,+0.000) (-0.5,+0) |
| sketch | cpolar(.3,-.4,-.2,.1) | 4 | 0.001 | nan | None | ns_cubic5 (nan) | (+0.300,-0.400,-0.200,+0.100) (+0.3,-0.4,-0.2,+0.1) |
| sketch | frob | 4 | 0.01 | 9.96e-03 | 1 | spec_p0.5 (27.6) | (+1.000) (+1) |
| sketch | sign | 4 | 0.01 | 1.01e-02 | 1 | row_norm (50.6) |  |
| sketch | row_norm | 4 | 0.01 | 1.00e-02 | 1 | cpolar_rowwhite (42.1) | (-0.500,+0.000) (-0.5,+0) |
| sketch | col_norm | 4 | 0.01 | 1.00e-02 | 1 | spec_p0.5 (26.2) | (+0.000,-0.500) (+0,-0.5) |
| sketch | polar | 4 | 0.01 | 1.00e-02 | 1 | ns_cubic15 (1) | (-0.000) (+0) |
| sketch | ns_quintic5 | 4 | 0.01 | 9.99e-03 | 2 | ns_quintic5_f32 (1) | (+3.445,-4.775,+2.031) (+3.44,-4.78,+2.03) |
| sketch | ns_quintic3 | 4 | 0.01 | 1.00e-02 | 1 | ns_quintic5 (5.65) |  |
| sketch | ns_quintic8 | 4 | 0.01 | 9.96e-03 | 1 | ns_cubic15 (17) |  |
| sketch | ns_cubic5 | 4 | 0.01 | 9.97e-03 | 1 | ns_cubic15 (10.4) |  |
| sketch | ns_cubic15 | 4 | 0.01 | 9.98e-03 | 2 | polar (1) |  |
| sketch | spec_p0.5 | 4 | 0.01 | 1.00e-02 | 1 | frob (22.2) | (+0.500) (+0.5) |
| sketch | dpre_-.25_-.25 | 4 | 0.01 | 1.00e-02 | 1 | cpolar_dshampoo (38.2) | (-0.250,-0.250) (-0.25,-0.25) |
| sketch | cpolar_rowwhite | 4 | 0.01 | 1.00e-02 | 1 | row_norm (36.3) | (+0.000,-0.500,+0.001,+0.000) (+0,-0.5,+0,+0) |
| sketch | cpolar_colwhite | 4 | 0.01 | 1.01e-02 | 1 | ns_cubic15 (9.38) | (-0.000,+0.000,-0.500,-0.000) (+0,+0,-0.5,+0) |
| sketch | cpolar_dshampoo | 4 | 0.01 | 9.97e-03 | 1 | dpre_-.25_-.25 (44) | (-0.000,-0.250,-0.250,-0.000) (+0,-0.25,-0.25,+0) |
| sketch | ns_quintic5_f32 | 4 | 0.01 | 9.92e-03 | 2 | ns_quintic5 (1) |  |
| sketch | ns_quintic5_bf16 | 4 | 0.01 | 1.01e-02 | 1 | ns_quintic5 (3.19) |  |
| sketch | apolar_-.5 | 4 | 0.01 | 1.00e-02 | 1 | ns_cubic15 (1.44) | (-0.501,+0.000) (-0.5,+0) |
| sketch | cpolar(.3,-.4,-.2,.1) | 4 | 0.01 | nan | None | ns_cubic5 (nan) | (+0.300,-0.400,-0.200,+0.100) (+0.3,-0.4,-0.2,+0.1) |
| sketch | frob | 16 | 1e-06 | 9.98e-07 | 1 | spec_p0.5 (3.2e+05) | (+1.000) (+1) |
| sketch | sign | 16 | 1e-06 | 1.00e-06 | 1 | row_norm (5.98e+05) |  |
| sketch | row_norm | 16 | 1e-06 | 9.99e-07 | 1 | dpre_-.25_-.25 (4.47e+05) | (-0.500,-0.000) (-0.5,+0) |
| sketch | col_norm | 16 | 1e-06 | 1.00e-06 | 1 | frob (2.86e+05) | (-0.000,-0.500) (+0,-0.5) |
| sketch | polar | 16 | 1e-06 | 9.99e-07 | 1 | apolar_-.5 (2.79e+04) | (-0.000) (+0) |
| sketch | ns_quintic5 | 16 | 1e-06 | 9.98e-07 | 1 | ns_quintic5_f32 (2.66) | (+3.444,-4.775,+2.032) (+3.44,-4.78,+2.03) |
| sketch | ns_quintic3 | 16 | 1e-06 | 1.01e-06 | 1 | ns_cubic5 (4.51e+05) |  |
| sketch | ns_quintic8 | 16 | 1e-06 | 9.94e-07 | 1 | ns_cubic15 (1.89e+05) |  |
| sketch | ns_cubic5 | 16 | 1e-06 | 1.00e-06 | 1 | spec_p0.5 (2.32e+05) |  |
| sketch | ns_cubic15 | 16 | 1e-06 | 1.00e-06 | 1 | polar (6.41e+04) |  |
| sketch | spec_p0.5 | 16 | 1e-06 | 9.98e-07 | 1 | ns_cubic5 (2.36e+05) | (+0.500) (+0.5) |
| sketch | dpre_-.25_-.25 | 16 | 1e-06 | 1.00e-06 | 1 | col_norm (4.5e+05) | (-0.250,-0.250) (-0.25,-0.25) |
| sketch | cpolar_rowwhite | 16 | 1e-06 | 9.99e-07 | 1 | cpolar_dshampoo (4.32e+05) | (-0.000,-0.500,+0.000,+0.000) (+0,-0.5,+0,+0) |
| sketch | cpolar_colwhite | 16 | 1e-06 | 9.97e-07 | 1 | apolar_-.5 (2.32e+05) | (-0.000,+0.000,-0.500,-0.000) (+0,+0,-0.5,+0) |
| sketch | cpolar_dshampoo | 16 | 1e-06 | 9.99e-07 | 1 | cpolar_colwhite (4.18e+05) | (-0.000,-0.250,-0.250,+0.000) (+0,-0.25,-0.25,+0) |
| sketch | ns_quintic5_f32 | 16 | 1e-06 | 9.95e-07 | 1 | ns_quintic5 (3.01) |  |
| sketch | ns_quintic5_bf16 | 16 | 1e-06 | 1.00e-06 | 1 | ns_quintic5_f32 (1.23e+05) |  |
| sketch | apolar_-.5 | 16 | 1e-06 | 1.00e-06 | 1 | polar (2.82e+04) | (-0.500,+0.000) (-0.5,+0) |
| sketch | cpolar(.3,-.4,-.2,.1) | 16 | 1e-06 | nan | None | ns_cubic15 (nan) | (+0.300,-0.400,-0.200,+0.100) (+0.3,-0.4,-0.2,+0.1) |
| sketch | frob | 16 | 0.001 | 9.98e-04 | 1 | spec_p0.5 (320) | (+1.000) (+1) |
| sketch | sign | 16 | 0.001 | 1.00e-03 | 1 | row_norm (598) |  |
| sketch | row_norm | 16 | 0.001 | 9.99e-04 | 1 | dpre_-.25_-.25 (447) | (-0.500,-0.000) (-0.5,+0) |
| sketch | col_norm | 16 | 0.001 | 1.00e-03 | 1 | frob (286) | (-0.000,-0.500) (+0,-0.5) |
| sketch | polar | 16 | 0.001 | 9.99e-04 | 1 | apolar_-.5 (28) | (+0.000) (+0) |
| sketch | ns_quintic5 | 16 | 0.001 | 9.98e-04 | 1 | ns_quintic5_f32 (1) | (+3.444,-4.775,+2.032) (+3.44,-4.78,+2.03) |
| sketch | ns_quintic3 | 16 | 0.001 | 1.01e-03 | 1 | ns_cubic5 (451) |  |
| sketch | ns_quintic8 | 16 | 0.001 | 9.94e-04 | 1 | ns_cubic15 (189) |  |
| sketch | ns_cubic5 | 16 | 0.001 | 1.00e-03 | 1 | spec_p0.5 (232) |  |
| sketch | ns_cubic15 | 16 | 0.001 | 1.00e-03 | 1 | polar (64.1) |  |
| sketch | spec_p0.5 | 16 | 0.001 | 9.98e-04 | 1 | ns_cubic5 (236) | (+0.500) (+0.5) |
| sketch | dpre_-.25_-.25 | 16 | 0.001 | 1.00e-03 | 1 | col_norm (450) | (-0.250,-0.250) (-0.25,-0.25) |
| sketch | cpolar_rowwhite | 16 | 0.001 | 9.99e-04 | 1 | cpolar_dshampoo (432) | (-0.000,-0.500,+0.000,+0.000) (+0,-0.5,+0,+0) |
| sketch | cpolar_colwhite | 16 | 0.001 | 9.97e-04 | 1 | apolar_-.5 (232) | (-0.000,+0.000,-0.500,-0.000) (+0,+0,-0.5,+0) |
| sketch | cpolar_dshampoo | 16 | 0.001 | 9.99e-04 | 1 | cpolar_colwhite (418) | (-0.000,-0.250,-0.250,+0.000) (+0,-0.25,-0.25,+0) |
| sketch | ns_quintic5_f32 | 16 | 0.001 | 9.95e-04 | 1 | ns_quintic5 (1) |  |
| sketch | ns_quintic5_bf16 | 16 | 0.001 | 1.00e-03 | 1 | ns_quintic5_f32 (123) |  |
| sketch | apolar_-.5 | 16 | 0.001 | 1.00e-03 | 1 | polar (28.2) | (-0.500,+0.000) (-0.5,+0) |
| sketch | cpolar(.3,-.4,-.2,.1) | 16 | 0.001 | nan | None | ns_cubic15 (nan) | (+0.300,-0.400,-0.200,+0.100) (+0.3,-0.4,-0.2,+0.1) |
| sketch | frob | 16 | 0.01 | 9.98e-03 | 1 | spec_p0.5 (32) | (+1.000) (+1) |
| sketch | sign | 16 | 0.01 | 1.00e-02 | 1 | row_norm (59.8) |  |
| sketch | row_norm | 16 | 0.01 | 9.99e-03 | 1 | dpre_-.25_-.25 (44.7) | (-0.500,-0.000) (-0.5,+0) |
| sketch | col_norm | 16 | 0.01 | 1.00e-02 | 1 | frob (28.6) | (-0.000,-0.500) (+0,-0.5) |
| sketch | polar | 16 | 0.01 | 9.99e-03 | 1 | apolar_-.5 (2.97) | (+0.000) (+0) |
| sketch | ns_quintic5 | 16 | 0.01 | 9.98e-03 | 2 | ns_quintic5_f32 (1) | (+3.444,-4.775,+2.032) (+3.44,-4.78,+2.03) |
| sketch | ns_quintic3 | 16 | 0.01 | 1.01e-02 | 1 | ns_cubic5 (45.1) |  |
| sketch | ns_quintic8 | 16 | 0.01 | 9.94e-03 | 1 | ns_cubic15 (18.9) |  |
| sketch | ns_cubic5 | 16 | 0.01 | 1.00e-02 | 1 | spec_p0.5 (23.2) |  |
| sketch | ns_cubic15 | 16 | 0.01 | 1.00e-02 | 1 | polar (6.49) |  |
| sketch | spec_p0.5 | 16 | 0.01 | 9.98e-03 | 1 | ns_cubic5 (23.6) | (+0.500) (+0.5) |
| sketch | dpre_-.25_-.25 | 16 | 0.01 | 1.00e-02 | 1 | col_norm (45) | (-0.250,-0.250) (-0.25,-0.25) |
| sketch | cpolar_rowwhite | 16 | 0.01 | 9.99e-03 | 1 | cpolar_dshampoo (43.2) | (-0.000,-0.500,+0.000,+0.000) (+0,-0.5,+0,+0) |
| sketch | cpolar_colwhite | 16 | 0.01 | 9.97e-03 | 1 | apolar_-.5 (23.2) | (-0.000,+0.000,-0.500,-0.000) (+0,+0,-0.5,+0) |
| sketch | cpolar_dshampoo | 16 | 0.01 | 9.99e-03 | 1 | cpolar_colwhite (41.8) | (-0.000,-0.250,-0.250,+0.000) (+0,-0.25,-0.25,+0) |
| sketch | ns_quintic5_f32 | 16 | 0.01 | 9.95e-03 | 1 | ns_quintic5 (1) |  |
| sketch | ns_quintic5_bf16 | 16 | 0.01 | 1.00e-02 | 1 | ns_quintic5_f32 (12.3) |  |
| sketch | apolar_-.5 | 16 | 0.01 | 1.00e-02 | 1 | polar (2.99) | (-0.500,+0.001) (-0.5,+0) |
| sketch | cpolar(.3,-.4,-.2,.1) | 16 | 0.01 | nan | None | ns_cubic15 (nan) | (+0.300,-0.400,-0.200,+0.100) (+0.3,-0.4,-0.2,+0.1) |
| sketch | frob | 64 | 1e-06 | 9.99e-07 | 1 | spec_p0.5 (3.21e+05) | (+1.000) (+1) |
| sketch | sign | 64 | 1e-06 | 9.99e-07 | 1 | row_norm (6e+05) |  |
| sketch | row_norm | 64 | 1e-06 | 9.99e-07 | 1 | dpre_-.25_-.25 (4.48e+05) | (-0.500,+0.000) (-0.5,+0) |
| sketch | col_norm | 64 | 1e-06 | 1.00e-06 | 1 | frob (3.36e+05) | (-0.000,-0.500) (+0,-0.5) |
| sketch | polar | 64 | 1e-06 | 1.00e-06 | 1 | apolar_-.5 (5.13e+04) | (+0.000) (+0) |
| sketch | ns_quintic5 | 64 | 1e-06 | 9.99e-07 | 1 | ns_quintic5_f32 (5.56) | (+3.445,-4.775,+2.032) (+3.44,-4.78,+2.03) |
| sketch | ns_quintic3 | 64 | 1e-06 | 9.99e-07 | 1 | spec_p0.5 (4.3e+05) |  |
| sketch | ns_quintic8 | 64 | 1e-06 | 1.00e-06 | 1 | polar (1.73e+05) |  |
| sketch | ns_cubic5 | 64 | 1e-06 | 1.00e-06 | 1 | spec_p0.5 (2.48e+05) |  |
| sketch | ns_cubic15 | 64 | 1e-06 | 1.00e-06 | 1 | ns_quintic5_f32 (1.71e+05) |  |
| sketch | spec_p0.5 | 64 | 1e-06 | 1.00e-06 | 1 | ns_cubic5 (2.71e+05) | (+0.500) (+0.5) |
| sketch | dpre_-.25_-.25 | 64 | 1e-06 | 1.00e-06 | 1 | col_norm (4.24e+05) | (-0.250,-0.250) (-0.25,-0.25) |
| sketch | cpolar_rowwhite | 64 | 1e-06 | 1.00e-06 | 1 | cpolar_dshampoo (4.32e+05) | (+0.000,-0.500,+0.000,+0.000) (+0,-0.5,+0,+0) |
| sketch | cpolar_colwhite | 64 | 1e-06 | 1.00e-06 | 1 | polar (2.47e+05) | (-0.000,+0.000,-0.500,+0.000) (+0,+0,-0.5,+0) |
| sketch | cpolar_dshampoo | 64 | 1e-06 | 1.00e-06 | 1 | polar (3.81e+05) | (-0.000,-0.250,-0.250,+0.000) (+0,-0.25,-0.25,+0) |
| sketch | ns_quintic5_f32 | 64 | 1e-06 | 1.00e-06 | 1 | ns_quintic5 (6.08) |  |
| sketch | ns_quintic5_bf16 | 64 | 1e-06 | 1.00e-06 | 1 | ns_quintic5_f32 (2.07e+05) |  |
| sketch | apolar_-.5 | 64 | 1e-06 | 9.99e-07 | 1 | polar (4.92e+04) | (-0.500,+0.000) (-0.5,+0) |
| sketch | cpolar(.3,-.4,-.2,.1) | 64 | 1e-06 | nan | None | polar (nan) | (+0.300,-0.400,-0.200,+0.100) (+0.3,-0.4,-0.2,+0.1) |
| sketch | frob | 64 | 0.001 | 9.99e-04 | 1 | spec_p0.5 (321) | (+1.000) (+1) |
| sketch | sign | 64 | 0.001 | 9.99e-04 | 1 | row_norm (600) |  |
| sketch | row_norm | 64 | 0.001 | 9.99e-04 | 1 | dpre_-.25_-.25 (448) | (-0.500,+0.000) (-0.5,+0) |
| sketch | col_norm | 64 | 0.001 | 1.00e-03 | 1 | frob (336) | (-0.000,-0.500) (+0,-0.5) |
| sketch | polar | 64 | 0.001 | 1.00e-03 | 1 | apolar_-.5 (51.3) | (+0.000) (+0) |
| sketch | ns_quintic5 | 64 | 0.001 | 9.99e-04 | 1 | ns_quintic5_f32 (1) | (+3.444,-4.775,+2.032) (+3.44,-4.78,+2.03) |
| sketch | ns_quintic3 | 64 | 0.001 | 9.99e-04 | 1 | spec_p0.5 (430) |  |
| sketch | ns_quintic8 | 64 | 0.001 | 1.00e-03 | 1 | polar (173) |  |
| sketch | ns_cubic5 | 64 | 0.001 | 1.00e-03 | 1 | spec_p0.5 (248) |  |
| sketch | ns_cubic15 | 64 | 0.001 | 1.00e-03 | 1 | ns_quintic5_f32 (171) |  |
| sketch | spec_p0.5 | 64 | 0.001 | 1.00e-03 | 1 | ns_cubic5 (271) | (+0.500) (+0.5) |
| sketch | dpre_-.25_-.25 | 64 | 0.001 | 1.00e-03 | 1 | col_norm (424) | (-0.250,-0.250) (-0.25,-0.25) |
| sketch | cpolar_rowwhite | 64 | 0.001 | 1.00e-03 | 1 | cpolar_dshampoo (432) | (+0.000,-0.500,-0.000,+0.000) (+0,-0.5,+0,+0) |
| sketch | cpolar_colwhite | 64 | 0.001 | 1.00e-03 | 1 | polar (247) | (-0.000,+0.000,-0.500,+0.000) (+0,+0,-0.5,+0) |
| sketch | cpolar_dshampoo | 64 | 0.001 | 1.00e-03 | 1 | polar (381) | (+0.000,-0.250,-0.250,+0.000) (+0,-0.25,-0.25,+0) |
| sketch | ns_quintic5_f32 | 64 | 0.001 | 1.00e-03 | 1 | ns_quintic5 (1) |  |
| sketch | ns_quintic5_bf16 | 64 | 0.001 | 1.00e-03 | 1 | ns_quintic5_f32 (207) |  |
| sketch | apolar_-.5 | 64 | 0.001 | 9.99e-04 | 1 | polar (49.2) | (-0.500,+0.000) (-0.5,+0) |
| sketch | cpolar(.3,-.4,-.2,.1) | 64 | 0.001 | nan | None | polar (nan) | (+0.300,-0.400,-0.200,+0.100) (+0.3,-0.4,-0.2,+0.1) |
| sketch | frob | 64 | 0.01 | 9.99e-03 | 1 | spec_p0.5 (32.1) | (+1.000) (+1) |
| sketch | sign | 64 | 0.01 | 9.99e-03 | 1 | row_norm (60) |  |
| sketch | row_norm | 64 | 0.01 | 9.99e-03 | 1 | dpre_-.25_-.25 (44.8) | (-0.500,+0.000) (-0.5,+0) |
| sketch | col_norm | 64 | 0.01 | 1.00e-02 | 1 | frob (33.6) | (-0.000,-0.500) (+0,-0.5) |
| sketch | polar | 64 | 0.01 | 1.00e-02 | 1 | apolar_-.5 (5.22) | (+0.000) (+0) |
| sketch | ns_quintic5 | 64 | 0.01 | 9.99e-03 | 2 | ns_quintic5_f32 (1) | (+3.444,-4.775,+2.032) (+3.44,-4.78,+2.03) |
| sketch | ns_quintic3 | 64 | 0.01 | 9.99e-03 | 1 | spec_p0.5 (43.1) |  |
| sketch | ns_quintic8 | 64 | 0.01 | 1.00e-02 | 1 | polar (17.3) |  |
| sketch | ns_cubic5 | 64 | 0.01 | 1.00e-02 | 1 | spec_p0.5 (24.8) |  |
| sketch | ns_cubic15 | 64 | 0.01 | 1.00e-02 | 1 | ns_quintic5_f32 (17.2) |  |
| sketch | spec_p0.5 | 64 | 0.01 | 1.00e-02 | 1 | ns_cubic5 (27.1) | (+0.500) (+0.5) |
| sketch | dpre_-.25_-.25 | 64 | 0.01 | 1.00e-02 | 1 | col_norm (42.4) | (-0.250,-0.250) (-0.25,-0.25) |
| sketch | cpolar_rowwhite | 64 | 0.01 | 1.00e-02 | 1 | cpolar_dshampoo (43.2) | (+0.000,-0.500,-0.000,+0.000) (+0,-0.5,+0,+0) |
| sketch | cpolar_colwhite | 64 | 0.01 | 1.00e-02 | 1 | polar (24.7) | (-0.000,+0.000,-0.500,+0.000) (+0,+0,-0.5,+0) |
| sketch | cpolar_dshampoo | 64 | 0.01 | 1.00e-02 | 1 | polar (38.2) | (+0.000,-0.250,-0.250,+0.000) (+0,-0.25,-0.25,+0) |
| sketch | ns_quintic5_f32 | 64 | 0.01 | 1.00e-02 | 1 | ns_quintic5 (1) |  |
| sketch | ns_quintic5_bf16 | 64 | 0.01 | 1.00e-02 | 1 | ns_quintic5_f32 (20.8) |  |
| sketch | apolar_-.5 | 64 | 0.01 | 9.99e-03 | 1 | polar (5.02) | (-0.500,+0.000) (-0.5,+0) |
| sketch | cpolar(.3,-.4,-.2,.1) | 64 | 0.01 | nan | None | polar (nan) | (+0.300,-0.400,-0.200,+0.100) (+0.3,-0.4,-0.2,+0.1) |
