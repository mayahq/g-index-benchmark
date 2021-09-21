# Towards a Measure of General Intelligence
## Introduction
This repository contains the code to measure [g-index](definitions.md) for an experiment as described in the [paper](https://www.example.com)

# Table of Contents
  * [Results](#results)
  * [Plots](#plots)
  * [Usage](#usage)
  * [Running the Simulations](#running-the-simulations)
    + [Preparation:](#preparation-)
    + [Using the Command Line](#using-the-command-line)
      - [Options](#options)
    + [Running the Simulations:](#running-the-simulations-)
    + [Request the data](#request-the-data)
  * [License](#license)
  * [Cite Us!](#cite-us)

## Results 
|    | Model Name   | \# Training Samples | Compute Used                                  | $\theta$       | g_index           |
|----|--------------|---------------------|-----------------------------------------------|----------------|-------------------|
| 1. | GPT2-345M    | 2560                | 127.530                                       | 0.697          | 7902.972          |
| 2. | GPT Neo-2.7B | 2560                | 8969.100                                      | 0.682          | 6421.049          |
| 3. | GPT2-1.5B    | 5120                | 5927.400                                      | 0.708          | 6390.314          |
| 4. | GPT2-1.5B    | 10240               | 11563.320                                     | 0.683          | 6006.261          |
| 5. | GPT2-774M    | 2560                | 1516.640                                      | 0.620          | 4872.334          |
| 6. | GPT Neo-2.7B | 1280                | 5063.380                                      | 0.582          | 4476.680          |
| 7. | GPT2-345M    | 1280                | 74.750                                        | 0.547          | 4399.190          |
| 8. | GPT2-774M    | 5120                | 2941.941                                      | 0.585          | 4070.117          |
<blockquote>
 Each model was trained for 30 epochs 
</blockquote>


## Running the Simulations
### Preparation:
1. Install the Python package requirements via the following command:
   ```bash
        pip install -r requirements.txt
    ```
2. 
3. After getting the data, verify you have the following copied to the root of the repository
    1. experiments folder
    2. templates folder
    3. lengths.json file

### Using the Command Line 
The command line offers various flags to reproduce the results as claimed in the paper.

#### Options
1. `-e`, `--experiment-dir` &nbsp;[Required,&nbsp; `String`]: Set the directory where the experiment files are stored
2. `-t` `--template_dir`  &nbsp;[Required,&nbsp; `String`]: Set the directory where the template files are stored
3. `-p` `--print_results` &nbsp;[Optional,&nbsp; `Bool`]: Set whether to print the metrics on the command line.
4. `-s` `--save_metrics` &nbsp; [Optional &nbsp; `Bool`]: Set whether to dump metrics to a JSON file.
5. `-rs` `--rum_simulation` &nbsp; [Optional &nbsp; `Bool`]: Set whether to calculate `g-index` using the simulated values.
6. `-sf` `--sim_json` &nbsp; [Optional &nbsp; `String`]: JSON file with values to run simulation in case of `run_simulation == True`. 
Alternatively, you can also run `python main.py -h` to see the available options

### Running the Simulations:
Calculating `g-index` using simulated values is supported, however only for studying the behaviour of the `g-index` w.r.t to various indices of the experiment. It might be possible that the simulated values of `g-index` might never be achievable through experiments. Following are the steps:
1. Open [values.json](assets/values.json) located under assets.
2. Set the values of the various Experiment Indices (refer to [definitions.md](definitions.md)) 
3. After setting the desired values, simply run 
    ```bash 
    python main.py -p True -s True -rs True -sf assets/values.json 
    ```
4. Results will be printed and saved under `results` folder which can be used for plotting.

### Request the data
You can send us a mail at [humans@mayahq.com](mailto:humans@mayahq.com) breifly describing your use case to get the data.

## License

## Cite Us!
```
Citing Details
```
