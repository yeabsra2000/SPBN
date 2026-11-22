import torch, torch.nn.functional as F, itertools
from spbn_core.core.layers import ProbabilisticBeliefLayer
from spbn_core.core.updates import LocalBayesianUpdater
from spbn_core.training.trainer import PredictiveCodingTrainer

B,IN,H,OUT=64,8,16,3
g=torch.Generator().manual_seed(3)
x=torch.randn(B,IN,generator=g)
A=torch.randn(H,IN,generator=g)/IN**0.5
B2=torch.randn(OUT,H,generator=g)/H**0.5
target=torch.tanh(x@A.t())@B2.t()

def run(iv,thr,lr_mu,lr_var,lam,steps=500):
    torch.manual_seed(0)
    l1=ProbabilisticBeliefLayer(IN,H,initial_var=iv,snr_threshold=thr)
    l2=ProbabilisticBeliefLayer(H,OUT,initial_var=iv,snr_threshold=thr)
    upd=LocalBayesianUpdater(lr_mu,lr_var,lam,var_min=0.05)
    tr=PredictiveCodingTrainer([l1,l2],upd,activation="tanh",threshold=thr)
    init=F.mse_loss(tr.predict(x),target).item()
    for s in range(steps):
        tr.train_step(x,target)
    fin=F.mse_loss(tr.predict(x),target).item()
    return init,fin,tr.compression_rate()

for iv in [0.05,0.1]:
  for thr in [0.2,0.5]:
    for lr in [5e-4,1e-3,2e-3]:
      i,f,c=run(iv,thr,lr,1e-3,1e-3)
      print("iv=%.2f thr=%.1f lr=%.0e -> %.4f -> %.4f (pruned %.0f%%)"%(iv,thr,lr,i,f,c*100))
