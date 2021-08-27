# Towards a Measure of General Intelligence
## Introduction
This repository contains the code to measure [g-index](definitions.md) for an experiment as described in the paper [here](https://www.example.com)

## Results 

<table>
<thead>
<tr>
<th>Model Name</th>
<th>Number of <br> Parameters</th>
<th> G Index</th>
</tr>
</thead>
<tbody>
<tr>
<td>OpenAI GPT2 Medium</td>
<td>345M</td>
<td>1</td>
</tr>
<tr>
<td>OpenAI GPT2 Large</td>
<td>774M</td>
<td>1</td>
</tr>
<tr>
<td>OpenAI GPT2 XLarge</td>
<td>1.5B</td>
<td>1</td>
</tr>
<tr>
<td>EleutherAI GPT Neo</td>
<td>2.7B</td>
<td>1</td>
</tr>
</tbody>
</table>
<blockquote>
 Each model was trained on 8000 samples and 30 epochs 
</blockquote>

## Plots
<!-- [Replace this with a useful plot](images/sample.png "Replace this with a useful plot") -->
<img src="images/sample.png" alt="Replace this with a useful plot" style="height: 400px; width:400px;"/>

## Usage
### Preparation:
1. Install the Python package requirements via the following command:
   ```bash
        pip install -r requirements.txt
    ```
2. Follow the instructions [here](#request-the-data) to request the data
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
4. Results will be printed and saved under results which can be used for plotting.

### Request the data
You can send us a mail at [humans@mayahq.com](mailto:humans@mayahq.com) breifly describing your use case to get the data.

## License

## Cite Us!
```
Citing Details
```
