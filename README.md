# ARRAY

Benchmark Datasets and Source Code of the Paper "ARRAY: Adaptive Double Feature Weighted Transfer Naive Bayes for Cross-Project Defect Prediction".

## Tips
(1) RQ1 folder include the results of all models on each of all 242 combinations of source and taget datasets in terms of F1, AUC, and MCC;

(2) RQ2 folder includes the source code of the used project.

## Experimental Environment
(1) MATLAB R2018b;

(2) CPU：AMD Ryzen 5 4600U 2.10GHz  RAM 16GB;

(3) Python3.6, weka-3.8.4

## Usage of ARRAY
- Step1: Download ARRAY-master;
- Step2: Open MATLAB, add the path of ARRAY-master and its all subfolders into MATLAB;
- Step3: Add the absolute path of weka.jar in the subfolder ARRAY-master/ARRAY/ into the classpath.txt of MATLAB;
- Step4: Open the script Experiment.m (in the subfolder ARRAY-master/ARRAY/) with MATLAB Editor;
- Step5: Click the run button in the manu of MATLAB Editor to run the Experiment.m; 

If you use this repository in your research, please consider cite our work:
```
@article{Tong2023ARRAY,
        title = {ARRAY: Adaptive triple feature-weighted transfer Naive Bayes for cross-project defect prediction},
        journal = {Journal of Systems and Software},
        volume = {202},
        pages = {111721},
        year = {2023},
        author = {Haonan Tong and Wei Lu and Weiwei Xing and Shihai Wang},
}
```

If you have any question about ARRAY, do not hesitate to contact me (hntong@bjtu.edu.cn) or describe your question in Issues.
