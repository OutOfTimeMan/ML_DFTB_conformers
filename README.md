# ML_DFTB_conformers
## Abstract:

Large organic molecules and biomolecules can adopt multiple conformations, with the occurrences determined by their relative energies. Identifying the energetically most favorable conformations is crucial, especially
when interpreting spectroscopic experiments conducted under cryogenic conditions. When the effects of irregular surrounding medium, such as noble gas matrices, on the vibrational properties of molecules become
important, semi-empirical (SE) quantum-chemical methods are often employed for computational simulations. Although SE methods are computationally more efficient than first-principle quantum-chemical methods,
they can be inaccurate in determining the energies of conformers in some molecules while displaying good accuracy in others. In this study, we employ a combination of advanced machine learning techniques, such as
graph neural networks, to identify molecules with the highest errors in the relative energies of conformers computed by the semi-empirical tight-binding method GFN1-xTB. The performance of three different machine
learning models is assessed by comparing their predicted errors with the actual errors in conformer energies obtained via the GFN1-xTB method. We further applied the ensemble machine-learning model to a larger collection of molecules from the ChEMBL database and identified a set of molecules as being challenging for the GFN1-xTB method. These molecules hold potential for further improvement of the GFN1-xTB method, showcasing the capability of machine learning models in identifying molecules that can challenge its physical model.

## Results:
![A challenging test for DFTB semiempirical quantum-chemical methods built  with AI](https://github.com/OutOfTimeMan/ML_DFTB_conformers/assets/87600707/08875620-49dd-43cf-a0ea-8fa52e532a5d)

![A challenging test for DFTB semiempirical quantum-chemical methods built  with AI (1)](https://github.com/OutOfTimeMan/ML_DFTB_conformers/assets/87600707/f4642f87-ed74-4fca-bf96-570a82bf86b1)

We were able to successfully use ML to identify molecules that are difficult for the GFN-1xTB semiempirical method.  This was achieved through the creation of two types of graph neural networks and the Gradient Boosting method.

The ensemble of the three methods exhibits better generalization ability and higher accuracy (as expected and confirmed in our case). (Accuracy of classification for ensemble of models is 80%).

Out of the 157,901 molecules from the ChEMBL molecular database, our ensemble of models identified 1,528 molecules as xTB-challenging, so we recommend them  for further testing and improvement of the physical model of the GFN1-xTB method.

## The found set of molecules is available here: 
https://drive.google.com/drive/folders/1VFWYgB_LwlH29QY4Kg0L-crEdsnQGWBq

## The article is available here:
https://fnt.ilt.kharkiv.ua/index.php/fnt/article/view/9169
