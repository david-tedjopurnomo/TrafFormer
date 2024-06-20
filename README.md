# TrafFormer: Long-Term Traffic Prediction Transformer Model 

<p align="center">
	<img src="https://github.com/david-tedjopurnomo/long_term_traffic_prediction/blob/main/figures/3-trafformer-1.png" width=70% height=70%>
</p>

A long-term traffic prediction (i.e., up to 24 hours) model using a Transformer. The model uses the encoder part of a Transformer combined with a time-day embedding module to encode temporal information. Short and long-term past traffic data are fed into the model, where these data are denoted with $X$ and $X'$ respectively. The full paper is available [here](https://arxiv.org/pdf/2302.12388). Recommended citation is:

<code>David Alexander Tedjopurnomo, Farhana Choudhury, A. K. Qin. "TrafFormer: A Transformer Model for Prediction Long-term Traffic." arXiv preprint arXiv:2302.12388 (2023). https://doi.org/10.48550/arXiv.2302.12388</code>

This experiment uses the METR-LA and PEMS-BAY datasets from the [DCRNN](https://github.com/liyaguang/DCRNN) paper. Experiment results on these datasets comparing TrafFormer some deep learning models and state-of-the-art models [DCRNN](https://github.com/liyaguang/DCRNN), [STGCN](https://github.com/VeritasYin/STGCN_IJCAI-18) and [GAMCN](https://github.com/alvinzhaowei/GAMCN) are provided below.

<table class="tg">
<caption>METR-LA results</caption>
<thead>
  <tr>
    <th class="tg-c3ow" rowspan="2">Method</th>
    <th class="tg-c3ow" colspan="3">6 hours</th>
    <th class="tg-c3ow" colspan="3">12 hours</th>
    <th class="tg-c3ow" colspan="3">18 hours</th>
    <th class="tg-c3ow" colspan="3">24 hours</th>
  </tr>
  <tr>
    <th class="tg-c3ow">RMSE</th>
    <th class="tg-c3ow">MAE</th>
    <th class="tg-c3ow">MAPE</th>
    <th class="tg-c3ow">RMSE</th>
    <th class="tg-c3ow">MAE</th>
    <th class="tg-c3ow">MAPE</th>
    <th class="tg-c3ow">RMSE</th>
    <th class="tg-c3ow">MAE</th>
    <th class="tg-c3ow">MAPE</th>
    <th class="tg-c3ow">RMSE</th>
    <th class="tg-c3ow">MAE</th>
    <th class="tg-c3ow">MAPE</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow">FNN</td>
    <td class="tg-c3ow">11.46</td>
    <td class="tg-c3ow">6.05</td>
    <td class="tg-c3ow">22.14%</td>
    <td class="tg-c3ow">11.52</td>
    <td class="tg-c3ow">6.08</td>
    <td class="tg-c3ow">22.46%</td>
    <td class="tg-c3ow">11.54</td>
    <td class="tg-c3ow">6.10</td>
    <td class="tg-c3ow">22.56%</td>
    <td class="tg-c3ow">11.46</td>
    <td class="tg-c3ow">6.09</td>
    <td class="tg-c3ow">22.36%</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Stacked GRU</td>
    <td class="tg-c3ow">10.70</td>
    <td class="tg-c3ow">5.68</td>
    <td class="tg-c3ow">19.72%</td>
    <td class="tg-c3ow">10.78</td>
    <td class="tg-c3ow">5.72</td>
    <td class="tg-c3ow">20.03%</td>
    <td class="tg-c3ow">10.77</td>
    <td class="tg-c3ow">5.73</td>
    <td class="tg-c3ow">20.05%</td>
    <td class="tg-c3ow">10.69</td>
    <td class="tg-c3ow">5.71</td>
    <td class="tg-c3ow">19.78%</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Seq2Seq LSTM</td>
    <td class="tg-c3ow">9.68</td>
    <td class="tg-c3ow">5.02</td>
    <td class="tg-c3ow">16.83%</td>
    <td class="tg-c3ow">9.74</td>
    <td class="tg-c3ow">5.06</td>
    <td class="tg-c3ow">17.09%</td>
    <td class="tg-c3ow">9.77</td>
    <td class="tg-c3ow">5.07</td>
    <td class="tg-c3ow">17.17%</td>
    <td class="tg-c3ow">9.72</td>
    <td class="tg-c3ow">5.06</td>
    <td class="tg-c3ow">17.05%</td>
  </tr>
  <tr>
    <td class="tg-c3ow">DCRNN</td>
    <td class="tg-c3ow">8.99</td>
    <td class="tg-c3ow">4.29</td>
    <td class="tg-7btt">13.20%</td>
    <td class="tg-c3ow">9.33</td>
    <td class="tg-c3ow">4.48</td>
    <td class="tg-c3ow">13.61%</td>
    <td class="tg-c3ow">9.66</td>
    <td class="tg-c3ow">4.64</td>
    <td class="tg-c3ow">13.98%</td>
    <td class="tg-c3ow">9.63</td>
    <td class="tg-c3ow">4.66</td>
    <td class="tg-c3ow">14.29%</td>
  </tr>
  <tr>
    <td class="tg-c3ow">STGCN</td>
    <td class="tg-c3ow">14.86</td>
    <td class="tg-c3ow">8.76</td>
    <td class="tg-c3ow">21.07%</td>
    <td class="tg-c3ow">15.88</td>
    <td class="tg-c3ow">9.31</td>
    <td class="tg-c3ow">22.24%</td>
    <td class="tg-c3ow">15.31</td>
    <td class="tg-c3ow">9.12</td>
    <td class="tg-c3ow">21.99%</td>
    <td class="tg-c3ow">13.60</td>
    <td class="tg-c3ow">7.83</td>
    <td class="tg-c3ow">21.29%</td>
  </tr>
  <tr>
    <td class="tg-c3ow">GAMCN</td>
    <td class="tg-c3ow">13.33</td>
    <td class="tg-c3ow">6.30</td>
    <td class="tg-c3ow">16.23%</td>
    <td class="tg-c3ow">12.52</td>
    <td class="tg-c3ow">5.85</td>
    <td class="tg-c3ow">15.90%</td>
    <td class="tg-c3ow">11.51</td>
    <td class="tg-c3ow">5.35</td>
    <td class="tg-c3ow">15.07%</td>
    <td class="tg-c3ow">9.61</td>
    <td class="tg-c3ow">4.60</td>
    <td class="tg-c3ow">13.84%</td>
  </tr>
  <tr>
    <td class="tg-7btt">TrafFormer</td>
    <td class="tg-7btt">8.47</td>
    <td class="tg-7btt">4.11</td>
    <td class="tg-c3ow">13.25%</td>
    <td class="tg-7btt">8.53</td>
    <td class="tg-7btt">4.14</td>
    <td class="tg-7btt">13.39%</td>
    <td class="tg-7btt">8.56</td>
    <td class="tg-7btt">4.16</td>
    <td class="tg-7btt">13.50%</td>
    <td class="tg-7btt">8.59</td>
    <td class="tg-7btt">4.18</td>
    <td class="tg-7btt">13.57%</td>
  </tr>
</tbody>
</table>

<table class="tg">
<caption>PEMS-BAY results</caption>
<thead>
  <tr>
    <th class="tg-c3ow" rowspan="2">Method</th>
    <th class="tg-c3ow" colspan="3">6 hours</th>
    <th class="tg-c3ow" colspan="3">12 hours</th>
    <th class="tg-c3ow" colspan="3">18 hours</th>
    <th class="tg-c3ow" colspan="3">24 hours</th>
  </tr>
  <tr>
    <th class="tg-c3ow">RMSE</th>
    <th class="tg-c3ow">MAE</th>
    <th class="tg-c3ow">MAPE</th>
    <th class="tg-c3ow">RMSE</th>
    <th class="tg-c3ow">MAE</th>
    <th class="tg-c3ow">MAPE</th>
    <th class="tg-c3ow">RMSE</th>
    <th class="tg-c3ow">MAE</th>
    <th class="tg-c3ow">MAPE</th>
    <th class="tg-c3ow">RMSE</th>
    <th class="tg-c3ow">MAE</th>
    <th class="tg-c3ow">MAPE</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow">FNN</td>
    <td class="tg-0pky">8.27</td>
    <td class="tg-0pky">3.83</td>
    <td class="tg-0pky">11.21%</td>
    <td class="tg-0pky">8.25</td>
    <td class="tg-0pky">3.83</td>
    <td class="tg-0pky">11.17%</td>
    <td class="tg-0pky">8.25</td>
    <td class="tg-0pky">3.83</td>
    <td class="tg-0pky">11.17%</td>
    <td class="tg-0pky">8.23</td>
    <td class="tg-0pky">3.84</td>
    <td class="tg-0pky">11.12%</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Stacked GRU</td>
    <td class="tg-0pky">6.20</td>
    <td class="tg-0pky">2.82</td>
    <td class="tg-0pky">7.17%</td>
    <td class="tg-0pky">6.20</td>
    <td class="tg-0pky">2.84</td>
    <td class="tg-0pky">7.19%</td>
    <td class="tg-0pky">6.20</td>
    <td class="tg-0pky">2.84</td>
    <td class="tg-0pky">7.19%</td>
    <td class="tg-0pky">6.20</td>
    <td class="tg-0pky">2.86</td>
    <td class="tg-0pky">7.20%</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Seq2Seq LSTM</td>
    <td class="tg-0pky">6.22</td>
    <td class="tg-0pky">2.83</td>
    <td class="tg-0pky">7.13%</td>
    <td class="tg-0pky">6.22</td>
    <td class="tg-0pky">2.83</td>
    <td class="tg-0pky">7.11%</td>
    <td class="tg-0pky">6.22</td>
    <td class="tg-0pky">2.83</td>
    <td class="tg-0pky">7.11%</td>
    <td class="tg-0pky">6.24</td>
    <td class="tg-0pky">2.85</td>
    <td class="tg-0pky">7.12%</td>
  </tr>
  <tr>
    <td class="tg-c3ow">DCRNN</td>
    <td class="tg-0pky">5.54</td>
    <td class="tg-0pky">2.45</td>
    <td class="tg-0pky">6.07%</td>
    <td class="tg-0pky">5.83</td>
    <td class="tg-0pky">2.54</td>
    <td class="tg-0pky">6.43%</td>
    <td class="tg-0pky">5.83</td>
    <td class="tg-0pky">2.54</td>
    <td class="tg-0pky">6.43%</td>
    <td class="tg-0pky">5.84</td>
    <td class="tg-0pky">2.58</td>
    <td class="tg-0pky">6.32%</td>
  </tr>
  <tr>
    <td class="tg-c3ow">STGCN</td>
    <td class="tg-0pky">5.76</td>
    <td class="tg-0pky">2.65</td>
    <td class="tg-0pky">6.30%</td>
    <td class="tg-0pky">6.18</td>
    <td class="tg-0pky">2.83</td>
    <td class="tg-0pky">6.86%</td>
    <td class="tg-0pky">6.18</td>
    <td class="tg-0pky">2.83</td>
    <td class="tg-0pky">6.86%</td>
    <td class="tg-0pky">6.81</td>
    <td class="tg-0pky">3.07</td>
    <td class="tg-0pky">7.8</td>
  </tr>
  <tr>
    <td class="tg-c3ow">GAMCN</td>
    <td class="tg-fymr">5.16</td>
    <td class="tg-fymr">2.30</td>
    <td class="tg-fymr">5.56%</td>
    <td class="tg-fymr">5.18</td>
    <td class="tg-fymr">2.31</td>
    <td class="tg-fymr">5.59%</td>
    <td class="tg-fymr">5.18</td>
    <td class="tg-fymr">2.31</td>
    <td class="tg-fymr">5.59%</td>
    <td class="tg-fymr">5.22</td>
    <td class="tg-fymr">2.34</td>
    <td class="tg-fymr">5.63%</td>
  </tr>
  <tr>
    <td class="tg-c3ow">TrafFormer</td>
    <td class="tg-0pky">5.47</td>
    <td class="tg-0pky">2.59</td>
    <td class="tg-0pky">6.15%</td>
    <td class="tg-0pky">5.43</td>
    <td class="tg-0pky">2.58</td>
    <td class="tg-0pky">6.09%</td>
    <td class="tg-0pky">5.43</td>
    <td class="tg-0pky">2.58</td>
    <td class="tg-0pky">6.09%</td>
    <td class="tg-0pky">5.46</td>
    <td class="tg-0pky">2.59</td>
    <td class="tg-0pky">6.14%</td>
  </tr>
</tbody>
</table>

## Requirements

```
numpy >= 1.23.4
pandas >= 1.5.2
pytables >= 3.7.0
scikit-learn >= 1.1.3
tensorflow-gpu == 2.4.1
```


## Getting Started

### Data Processor

The data processor script is modified from the [DCRNN](https://github.com/liyaguang/DCRNN/blob/master/scripts/generate_training_data.py) paper. This is done to minimize the variation from the DCRNN experiment so that the comparison will be as fair as possible. I modified this script to change the future prediction horizon and to add the long-term past data as the input. 

Before running the script, download the original ```metr-la.h5``` and ```pems-bay.h5``` datasets from this [link](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX), and place the files in a directory called ```data```. 

Running the script and generating the processed data use commands similar to DCRNN:

```
# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python -m data_processor.generate_training_data_dayhour --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python -m data_processor.generate_training_data_dayhour --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5
```

### Running TrafFormer 

All the arguments to the TrafFormer script is provided using an input .ini file. Two default ones, ```arg_pems.ini``` and ```arg_metr.ini``` are provided. The variable names in the file are case sensitive. The arguments and their usage are provided below: 

```
[INPUT]
InputTrainPath  = <String. Path to the input training data .npz file>
InputValPath    = <String. Path to the input validation data .npz file>
InputTestPath   = <String. Path to the input testing data .npz file>
InputModelPath  = <String. Path to the trained model. If you're training a new model, keep this blank>

[OUTPUT]
OutputBasePath  = <String. Path to the directory where the trained model and the results will be output to>

[MODEL]
ModelName       = <String. Name of model variation. The default value is ``trafformer_full"
HeadSize        = <Integer. Attention head size>
EmbedSize       = <Integer. Size of the day and hour embedding>
NumHeads        = <Integer. Number of attention heads>
FFDim           = <Integer. The number of feedforward filters>
NumTrfBlocks    = <Integer. The number of transformer blocks>
MLPUnits        = <Array of integers. Size of the MLP layers before the output layer>
Dropout         = <Integer. Dropout ratio for the Transformer part of the model>
MLPDropout      = <Integer. Dropout ratio for the MLP part of the model> 

[RESOURCE]
GPU             = <Integer. Used to select which GPU to use. Keep the value as 0 for single-GPU machines>

[TRAINING]
DoTraining      = <Boolean. Whether or not to train the model or not>
BatchSize       = <Integer. Batch size>
NumEpochs       = <Integer. Maximum number of epochs to traing the model>

[TESTING]
DoTesting       = <Boolean. Whether or not to test the model> 
```

Once the training and testing are done, the directory provided in ```OutputBasePath``` will be populated with four files: 

* ```<filename>.ini``` A copy of the provided .ini file for documentation purposes.
* ```checkpoiint.h5``` Checkpoint of the trained models' weights (i.e., not the full model, just the weights).
* ```results.txt``` Testing results.
* ```training.log``` Log of the model training.
