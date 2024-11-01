# %%

import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from utils import trim_wavelengths, train, train_lbfgs
# from overlapping_gp import OverlappingGP
from src.cogp import COGP
import numpy as np

torch.set_default_dtype(torch.float64)




msep = []
pls_msep = []


n_unobs_conc = 4

model = COGP(concs.shape[1], n_unobs_conc,  wavelengths_filtered.shape[0], torch.linspace(temps.min(), temps.max(), 2).reshape(-1,1), 
                    # beta=[2]*(concs.shape[1] + n_unobs_conc), 
                    #  sigma2=spectra.var(),
                    # sigma2_s=[spectra.var()/concs.std()]*(concs.shape[1] + n_unobs_conc)
                    )
loss = train(model, training_targets=train_spectra, test_targets=test_spectra, training_inputs=train_temps, test_inputs=test_temps, training_weights= train_concs, steps=200, lr = 1e-1)
loss.extend(train_lbfgs(model, training_targets=train_spectra, test_targets=test_spectra, training_inputs=train_temps, test_inputs=test_temps, training_weights= train_concs, steps=10))

# %%
with torch.no_grad():
    plt.figure()
    plt.plot(loss)
    plt.show()

    # num_samples = 10
    num_samples = 1000
    R_est = model.get_R_estimate_samples(test_spectra, test_temps, num_samples)
    plt.figure()
    for i in range(num_samples-1):
        plt.scatter(test_concs[:,0], R_est[i,:,0 ], c = test_temps, alpha=0.01,  )
    plt.scatter(test_concs[:,0], R_est[-1,:,0 ], c = test_temps, alpha=0.5,  label = "estimates for concentration")
        # plt.plot(range(R_est.shape[1]), temp_test, c = "red", alpha=1 )
    plt.plot(test_concs[:,0], test_concs[:,0])
    plt.legend()
    plt.savefig("figs/Lovastatin/0.5_test_pred.pdf")
    plt.show()

pls = PLSRegression(3)
pls.fit(train_spectra, train_concs)
y_hat = pls.predict(test_spectra)

plt.figure()
plt.scatter(test_concs[:,0], y_hat[:,0])
plt.plot(test_concs[:,0], test_concs[:,0])
plt.savefig("figs/Lovastatin/0.5_test_pred_pls.pdf")
plt.show()

msep.append(((R_est.mean(dim=0)[:,0] - test_concs[:,0])**2).mean())
pls_msep.append(((y_hat[:,0] - test_concs[:,0].numpy())**2).mean())


print(model.beta, model.sigma_s2, model.sigma2)
# %%

print("pls: ", np.mean(pls_msep))
print("gp:  ", np.mean(msep))

# for name, param in model.named_parameters():
#     print(name, param)


plt.figure()
plt.plot(pls_msep)
plt.plot(msep)
plt.show()
# %%
