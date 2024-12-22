# Optimal Transport for Domain Adaptation through Gaussian Mixture Models

This is the official repository for the paper [Optimal Transport for Domain Adaptation through Gaussian Mixture Models](https://openreview.net/forum?id=DCAeXwLenB), accepted in TMLR. Our paper uses the GMM-OTDA framework of (Delon and Desolneux, 2020) for domain adaptation, through 2 strategies,

- Mapping estimation, which maps points in the source domain towards the target domain using the GMMs,
- Label propagation, which estimates labels for the target domain GMM components.

You can run our code using,

```
python visda.py --base_path=PATH_TO_DATA --features="vit" --clusters_per_class="4" --reg_e=0.1
```

# Citation

```
@article{
    montesuma2024optimal,
    title={Optimal Transport for Domain Adaptation through Gaussian Mixture Models},
    author={Montesuma, Eduardo Fernandes and Mboula, Fred Maurice Ngol{\`e} and Souloumiac, Antoine},
    journal={Transactions on Machine Learning Research},
    year={2024},
    url={https://openreview.net/forum?id=DCAeXwLenB},
    note={Under review}
}
```

# References

- Delon, J., & Desolneux, A. (2020). A Wasserstein-type distance in the space of Gaussian mixture models. SIAM Journal on Imaging Sciences, 13(2), 936-970.