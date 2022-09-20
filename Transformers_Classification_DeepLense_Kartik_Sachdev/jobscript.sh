#!<SHEBANG>

# ask for eight tasks
#SBATCH --ntasks=8
 
# Ask for one node, use several nodes in case you need additional resources
#SBATCH --nodes=6
 
# ask for less than 4 GB memory per task=MPI rank
#SBATCH --mem-per-cpu=3900M   #M is the default and can therefore be omitted, but could also be K(ilo)|G(iga)|T(era)
 
# name the job
#SBATCH --job-name=DEEPLENSE

# request two gpus per node
#SBATCH --gres=gpu:pascal:1 # volta pascal

# declare the merged STDOUT/STDERR file
#SBATCH --output=output.%J.txt

#SBATCH --time=04:00:00

#SBATCH --mail-user=<email>
#SBATCH --mail-type=ALL 

### beginning of executable commands
### Change to working directory
cd ${DEEPLENSE_DIR}

### Execute your application
python3 -u main.py --num_workers 20 --dataset_name Model_III --train_config TwinsSVT --cuda