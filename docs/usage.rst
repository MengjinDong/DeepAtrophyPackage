**********************************************
DeepAtrophy Package Usage
**********************************************

This is a step-by-step guide on how to use DeepAtrophy. All example code to execucate the deepatrophy package is located in folder ``/deepatrophy/tools``. Illustrations of these scripts are displayed here.

Requirements
============
* deepatrophy package installed.
* A directory containing dataset images in pyramid format. We will assume that the path to this directory on your Linux machine is ``/deepatrophy/dataset``.


Preprocessing your Dataset
=============================

Neck trimming
----------------



Longitudinal alignment (rigid registration)
--------------------------------------------




Preparing your Dataset
=============================

Dataset can be split into need to be structured in the following way:

they need to be organized into csv files with the following columns:

- ``image_path``: path to the image file.
- ``mask_path``: path to the mask file.
- ``subject_id``: unique identifier for the subject.
- ``timepoint``: timepoint of the image.
- ``group``: group of the subject.
- ``age``: age of the subject.
- ``sex``:
- ``label``: label of the image.


We now have a train csv file and a test csv file. 


Training the model
=============================

We can now train the model using the following command:





Test the model
=============================

We can now test the model using the following command:




Analysis of the results
=============================

After testing the model, we obtain scores for each Longitudinal image pair in the test set. We can now generate Predicted Interscan Interval (PII) scores for each image pair using linear regression. 






We can now generate a scatter plot of the predicted vs actual PII scores.













