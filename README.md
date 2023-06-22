# Graph2Seq
This package contains the implementation of Graph2Seq paper in pytorch. 

Requirements

torch </br>
torch_geometric==2.3.1 </br>
torch_sparse </br>
networkx


running the model....


execute ./run.sh or python main.py

Configuration and hyperparameters can be obtained from configure.py file. 


this model has been evaluated on Shortest Path (synthetic dataset). It shows on average 99.97% accuracy compared 99.3% baseline results. For results comparison and more details, please see the report in pdf. 
