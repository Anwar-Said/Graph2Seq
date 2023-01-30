dataset="no-cycle"
seed="123"
epochs="50"
models="ResGatedGraphConv GCNConv SAGEConv GraphConv"
main="main.py"
for model in $models
do
    python $main --gnn $model --dataset $dataset --epochs $epochs --seed $seed
done