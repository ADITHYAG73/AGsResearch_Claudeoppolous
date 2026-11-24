"""fs_spec.py -- spectral (float64) transfer-function candidates for FIRSTSTEP-01."""
import numpy as np
A_PATH='/home/user/AGsResearch_Claudeoppolous/dimuon/runs/FS_cache/transfer_raw.npy'
# columns: r, s/||G||_F, s/s_max, d=-diag(P^T U Q), d_top, ||U||_F, d_out
MUON=[(3.4445,-4.7750,2.0315)]
# Polar Express (Amsel et al. 2025) schedule, written from memory; the fit below re-derives it.
PE_MEM=[(8.28721201814563,-23.595886519098837,17.300387312530933),
 (4.107059111542203,-2.9478499167379106,0.5448431082926601),
 (3.9486908534822946,-2.908902115962949,0.5518191394370137),
 (3.3184196573706015,-2.488488024314874,0.51004894012372),
 (2.300652019954817,-1.6689039845747493,0.4188073119525673),
 (1.891301407787398,-1.2679958271945868,0.37680408948524835),
 (1.8750014808534479,-1.2500016453999487,0.3750001645474248),
 (1.875,-1.25,0.375)]
def safety(coefs,sf=1.01,last_exempt=False):
    out=[(a/sf,b/sf**3,c/sf**5) for a,b,c in coefs]
    if last_exempt: out[-1]=coefs[-1]
    return out
def compose(x,coefs,k=1.0):
    y=np.asarray(x,float)*k
    for a,b,c in coefs: y=a*y+b*y**3+c*y**5
    return y
def load_curve():
    A=np.load(A_PATH); mid=np.cumsum(A[:,2]==1.0)-1; return A,mid
def score(fx,A,mid):
    nrm=np.sqrt(np.bincount(mid,fx**2)); p=fx/nrm[mid]*A[:,5]
    e=(p-A[:,3])/A[:,5]; out={}
    for r in [2,4,8,16,32,64]:
        m=A[:,0]==r; out[r]=np.sqrt(np.mean(e[m]**2))
    return out
