# Solving Optimal Control Problems using Soft Actor Critic 

The open-source implementation of the algorithms and the optimal control problems 

The following algorithms are implemented:


- Soft Actor Critic (SAC)
- Normalized Advantage Functions (NAF)
- Deep Deterministic Policy Gradient (DDPG)

The following optimal control problems are considered:    

- Van der Pol oscillator
- Pendulum
- Dubins Car
- Target Problem

 ## Requirements 
 
For training and evaluating described models, you will need python 3.6. To install requirements:    
    
```    
pip install -r requirements.txt    
```
    
## Training    
To train, run this command:    
    
```    
python train.py --config <path to config file>  
```

For example:
```    
python train.py --config .\configs\pendulum\naf.json
```
    
### **Training config file structure:** 

The training configuration file is presented as a json file with 3 required components - *environment*, *model*, *train_settings*.  
Each of these blocks has its own fields, presented in the tables below:  
  
#### environment's fields:  
  
| Parameter name| Type | example | Description |    
|-----------|------------|---------|-------------|    
| env_name|string| dubins-car| Optimal control problem to solve    
|dt| float  | 0.1        | Discretization step of continuous environment  

Possible *env_name* values:  
- *target-problem*
- *van-der-pol*  
- *pendulum*
- *dubins-car*  
  
#### model's fields:   

| Parameter name| Type | example | Description |    
|-----------|------------|---------|-------------|    
| model_name|string| naf| One of the algorithms, described in article    
|lr| float  | 0.001        | Learning rate  
|gamma| float  | 1        |Reward discount rate
|tau| float  | 0.01        | Smoothing parameter 

#### train_settings fields:

| Parameter name| Type | example | Description |    
|-----------|------------|---------|-------------|    
|epoch_num|int| 1000|  Number of training epochs   
|batch_size| int| 128        | Batch size  
|gamma| float  | 1        |Reward discount rate  
|render| boolean  | false        | Is need to visualize the environment during training  
|random_seed| int| 0        | Random seed to fix stochastic effects  
|save_rewards_path| path| \path\to\file       |   Path to save training reward history in numpy array format  
|save_model_path| path| \path\to\file       |   Path to save trained agent  
|save_plot_path| path| \path\to\file       |   Path to save training reward history plot  
  
#### train_config.json example:

```
{
  "environment": {
    "env_name": "pendulum",
    "dt": 0.1
  },
  "model": {
    "model_name": "naf",
    "lr": 0.001,
    "gamma": 1,
	"tau":0.01
  },
  "learning": {
    "epoch_num": 1000,
    "batch_size": 128,
    "render": false,
    "random_seed": 0,
    "save_rewards_path": "./data/pendulum/naf_rewards",
    "save_model_path": "./data/pendulum/naf_model",
    "save_plot_path": "./data/pendulum/naf_plot"
  }
}
```    
 > You can find prepared config files for all environments in folder **/configs**.  
## Evaluation    
To evaluate pre-trained model, run:    
    
```  
python eval.py --config <path to config file>    
```
For example:

```  
python eval.py --config .\configs\eval_config.json
```  


This script prints to the console all the states of the environment during the evaluation and outputs the final score.    
  #### **Evaluation config file  structure:**   
  The configuration file is presented as a json file with 3 required params - *environment*, *checkpoint*, *random_seed*.  
    
  
    
| Parameter name | Type | example | Description |    
|-----------|------------|---------|-------------|    
| environment|json |{"env_name": "dubins-car",  "dt": 0.1 } | the same object as in the training section  
|model    |path |  \path\to\checkpoint\file               | Path to pre-trained model  
|random_seed|int|  0               | Random seed to fix stochastic effects  
    
> Note that you can only use the model for the task on which it was trained    

#### eval_config.json example:

```
{    
  "environment": {    
     "env_name": "dubins-car",      
     "dt": 0.1    
 },  
  "model": "./data/pendulum/naf_model",
  "random_seed": 0      
}   
```    
