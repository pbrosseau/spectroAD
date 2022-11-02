# spectroAD

spectroAD, or Spectroscopy Anomaly Detection, is a set of tools for detection anomalies in spectroscopic data.

spectroAD was originally developed for detecting outliers in two-dimensional electronic spectroscopy (2DES) data, though these tools could easily be applied to any spectroscopic data.

-----------------------------------

The identification of outliers and data drift in high dimensional datasets is a complex problem as the large number of variables can dilute the variation between datasets. The challenges associated with analysing high dimensional datasets have been collectively referred to as the "curse of dimensionality" [1]. A typical solution is to reduce the dimensionality of the dataset by reducing the data to an orthonormal basis set. Individual experimental repetitions of a dataset can then be reconstructed as a linear combination of a finite number of basis vectors. Variation between experimental repetitions can be intuitively visualized in terms of two or three basis vectors, rather than the entire datasets. Principal components analysis (PCA) [2] is a common dimensionality reduction technique in chemometrics [3] and it has recently been used to identify defects in 2D materials [4] and reject outliers in magnetic resonance images (MRI) [5]. In PCA, the "principal components" are eigenvectors of the covariance matrix and form a basis set for reconstructing the data. In dimension reduction, the principal components used to describe the data are those which account for the greatest variance between samples or experimental repetitions.

The dissimilarity of the experimental repetitions can be quantified by measuring the distance between points in the principal components basis. This distance can be measured in a basic geometrical sense, by taking the difference between the coordinates. However, when there is significant covariance in a dataset the geometrical distance may not be meaningful and the Mahalanobis distance between points can give a more complete picture of the variation over the course of an experiment [3]. The Mahalanobis distance, M, is defined as:

M = sqrt((x - x_bar)S^-1(x-x_bar)^-1),

where x is a row vector representing a single dataset, x_bar is the mean row vector and $S$ is the covariance matrix [3]. The Mahalanobis distance therefore scales the distance between points by the covariance.

-----------------------------------

References:

1) A. Zimek, E. Schubert, and H. P. Kriegel, “A survey on unsupervised outlier detectionin high-dimensional numerical data,” Statistical Analysis and Data Mining 5, 363–387 (2012).
2) D. J. Bartholomew, “Principal components analysis,” International Encyclopedia of Education pp. 374–377 (2010).
3) R. G. Brereton, “The mahalanobis distance and its relationship to principal component scores,” Journal of Chemometrics 29, 143–145 (2015).
4) M. Amjadipour, J. Bradford, N. Zebardastan, P. V. Kolesnichenko, Q. Zhang,  C. Zheng, and M. S. Fuhrer, “Machine Learning : Science and Technology Multidimensional analysis of excitonic spectra of monolayers of tungsten disulphide : toward computer-aided identification of structural and environmental perturbations of 2D materials,” Machine Learning: Science and Technology 2, 025021 (2021).
5) A. F. Mejia, M. B. Nebel, A. Eloyan, B. Caffo, and M. A. Lindquist, “PCA leverage: Outlier detection for high-dimensional functional magnetic resonance imaging data,” Biostatistics 18, 521–536 (2017).
