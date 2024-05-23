# Face Presentation Attacks: Detection

**Ensemble learning and domain adaptation to improve the accuracy of face presentation attack detection**

## Motivation

The convenience and high accuracy of face recognition technology have led to its widespread adoption in everyday interactive tasks. However, face spoofing poses a significant vulnerability to face recognition systems due to its capacity to deceive and manipulate the technology.

The goal of this project is to explore different deep learning architectures (mainly Transformers and CNNs), the stacking ensemble learning technique, as well as data augmentation techniques, to improve the performance against spoofing attacks.

## Methodology

To evaluate the performance, OULU-NPU (denoted as O), Idiap Replay-Attack (denoted as I), MSU Mobile Face Spoofing (denoted as M), and CASIA (denoted as C) databases are utilized in our work. The Half Total Error Rate (HTER) is reported to provide an overall measure of the error rate in biometric authentication systems, balancing both false acceptances and false rejections.

### Preprocessing

The initial datasets consist of videos, from which a subset of frames needed to be extracted (5 per video). The face from each frame is then cropped out according to MTCNN predictions. Data augmentation is then applied: flip, color jitter, resize, normalization, and random rotation to a fixed angle.

### Training

The four datasets are employed as follows: three for training the model, from which we can split the training set and the validation set, and the fourth test set for model evaluation. This enhances domain adaptation. Consequently, four models are analyzed, each representing a unique combination: O&C&I to M, O&M&I to C, O&C&M to I, and I&C&M to O. Moreover, we consider also a combined cross-database evaluation: M&I to C, and M&I to O.

During the training process, we employ two important metrics to assess the model’s performance. The first metric is the Equal Error Rate (EER), which is computed while our model is trained on the validation set. The EER helps us gauge how well our model is performing in terms of achieving a balance between false positives and false negatives.

The second metric we employ is the Half Total Error Rate (HTER), which is exclusively computed on the test set. The HTER is a critical indicator of our model’s overall accuracy and its effectiveness in real-world scenarios. The results also consider the Area Under the ROC Curve (AUC). The AUC allows us to evaluate the same model’s performance across different thresholds.

## Repository Content

- **create_dataset**: Jupyter notebook to create the training dataset
- **face-presentation-attacks**: base models training
- **stacking ensemble**: meta model training
- **streaming**: inference over the webcam
- **report**: full PDF report
