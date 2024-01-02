![Screenshot from 2024-01-02 19-36-47](https://github.com/ljfanxi/lane-detection-RANSAC/assets/61730377/ad3e35e6-0f29-4c4d-a0f5-b70c2f90f352)# RANSAC Non-Linear Regression
1. Feature extraction 
Using HSV color identification
Extract Color: yellow and white
2. HoughLine to get coordinate points
3. Sklearn for analysis for non-linear regression
![Screenshot from 2024-01-02 19-35-12](https://github.com/ljfanxi/lane-detection-RANSAC/assets/61730377/12b03c71-519e-43f1-bb9a-4ebd7bff6539)
![Screenshot from 2024-01-02 19-36-47](https://github.com/ljfanxi/lane-detection-RANSAC/assets/61730377/17af1e5f-7ec7-4dd7-80c4-3f4981913dc0)

pros:
1. still can detect even road has shades.
2. .. even changes texture of road that turn into white.
3. .. even road lane is broken.

Unexperimented Bird Eye View yet. 
