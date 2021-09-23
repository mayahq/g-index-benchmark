# Towards a Measure of General Intelligence

## Table of Contents
  - [Introduction](#introduction)
  - [Results](#results)
    - [Guided Tour](#guided-tour)

## Introduction
This repository contains the code to measure [g-index](definitions.md) for an experiment as described in the [paper](https://www.example.com)

## Results 
|     | Model Name   | \# Training Samples | Compute Used | $\theta$ | g_index  |
| --- | ------------ | ------------------- | ------------ | -------- | -------- |
| 1.  | GPT2-345M    | 2560                | 127.530      | 0.697    | 7902.972 |
| 2.  | GPT Neo-2.7B | 2560                | 8969.100     | 0.682    | 6421.049 |
| 3.  | GPT2-1.5B    | 5120                | 5927.400     | 0.708    | 6390.314 |
| 4.  | GPT2-1.5B    | 10240               | 11563.320    | 0.683    | 6006.261 |
| 5.  | GPT2-774M    | 2560                | 1516.640     | 0.620    | 4872.334 |
| 6.  | GPT Neo-2.7B | 1280                | 5063.380     | 0.582    | 4476.680 |
| 7.  | GPT2-345M    | 1280                | 74.750       | 0.547    | 4399.190 |
| 8.  | GPT2-774M    | 5120                | 2941.941     | 0.585    | 4070.117 |
<blockquote>
 Each model was trained for 30 epochs 
</blockquote><br>

## Guided Tour 
Following are the instructions to reproduce the results in the paper.<br>
1. create a new conda/virtual environment 
2. install project requirements with
   ```bash
          pip install -r requirements.txt
      ```
3. Start Jupyter Lab with 
    ``` bash 
        jupyter lab
    ```
4. open `Tutorials.ipynb` located under `notebooks` and follow the instructions.
### Request the data
You can send us a mail at [humans@mayahq.com](mailto:humans@mayahq.com) breifly describing your use case to get the data.

## License

## Cite Us!
```
Citing Details
```
