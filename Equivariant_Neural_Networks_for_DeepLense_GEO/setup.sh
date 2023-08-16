pip install -r requirements.txt
sudo /opt/conda/bin/jupyter nbconvert --execute $1 --to notebook --inplace
sudo /opt/conda/bin/jupyter notebook --ip='0.0.0.0' --allow-root