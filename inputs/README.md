<h1 align="center"> Input files for training and testing the models </h1>

This folder contains six sub-folders corresponding to different VLL masses used for training and testing the DNN and SOM. A total of 200,000 events were generated for m<sub>&ell;</sub> = 500, 1000, 1500 GeV, out of which 12,000 events correspond to 3&ell; final state (&ell; = _e_, _&mu;_). For m<sub>&ell;</sub> = 750, 2500 GeV as well as _WZ_, we generate 50,000 events, and generate 100,000 events for _ttZ_. 

Each of folders `M500`, `M1000`, `M1500` contains three text files as: 9500 events used for training the DNN and SOM, 2000 events for testing the trained models, and 500 from which events will be sampled for the use cases. The folders `M750` and `M2500` contain one text file with 500 events which will be used to test the trained models. 

In the folder `SM`, for:

- DNN: The text files `TTZ_train` and `WZ_train` contain 4750 events each used for training DNN. The files `TTZ_test` and `WZ_test` contain 1000 events for testing the DNN. The events for the cases are sampled from `TTZ_sample500` and `WZ_sample500`.

- SOM: The text files `TTZ_1k_test` and `WZ_1k_test` contain 1000 events used for testing the SOM. (**Note that** SOM is not trained on SM processes). The events for the cases are sampled from `TTZ_500Ev_cases` and `WZ_500Ev_cases`.

