## Datasets' particulars

Server Machine Dataset (SMD). SMD is collected from a large Internet company and published on GitHub, whose observations are equally-spaced 1 minute apart. It includes time series of 28 machines distributed in three clusters. Each machine is pre-divided into a training set and a test set. Both of them contain 38 metrics. Anomalies in the test set are labeled by domain experts.

Secure Water Treatment (SWaT). SWaT is collected from a real-world water treatment plant, recording 11 days of continuous operation, 7 days under normal operation and 4 days with attack scenarios. Metrics are collected by 51 sensors and actuators. As two versions of the training set are available on the official website, we select the latest Version 1.

Water Distribution (WADI). WADI is an extension of the SWaT testbed with 123 sensors and actuators. Data under 14 days of normal operations are recorded as the training set, and data under attacks within 2 days are recorded as the test set. We use the latest CSV file "WADI_14days_new.csv" given by the official website as the training set, which removes readings affected by certain unstable periods.

## Preprocessing details
As the range of each metric in SMD has been limited to 0-1, we skip the data preprocessing of this dataset. For SWaT and WADI whose readings range from 0.01 to 1000, we apply MinMax Scaler to limit the value of the training set to 0-1. The maximum and the minimum value of the training set are further used as criteria to normalize the test set. Time series are then clipped to a proper scale. Four columns containing illegal values are dropped in WADI. To eliminate the effect of the unstable cold start phase of the system, we discard the first 21600 points of the training set of SWaT and WADI like MAD-GAN has done.
