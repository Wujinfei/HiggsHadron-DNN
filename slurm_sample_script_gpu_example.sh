#! /bin/bash
  
######## Part 1 #########
# Script parameters     #
#########################
 
# Specify the partition name from which resources will be allocated, mandatory option
#SBATCH --partition=gpu
 
# Specify the QOS, mandatory option
#SBATCH --qos=normal
 
# Specify which group you belong to, mandatory option
# This is for the accounting, so if you belong to many group,
# write the experiment which will pay for your resource consumption
#SBATCH --account=mlgpu
  
# Specify your job name, optional option, but strongly recommand to specify some name
#SBATCH --job-name=py3_AK  ##change to your job name
  
# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1

# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/bes/mlgpu/yuancc/ttHCPNewStrategy/ttH_multilepton/keras-DNN/slurmoutput/slurm_%u_%x_%j.out ##change to your output directory
  
# Specify memory to use, or slurm will allocate all available memory in MB
#SBATCH --mem-per-cpu=30000
  
# Specify how many GPU cards to use
#SBATCH --gres=gpu:v100:1
    
######## Part 2 ######
# Script workload    #
######################
  
# Replace the following lines with your real workload
########################################
cd /hpcfs/bes/mlgpu/yuancc/ttHCPNewStrategy/ttH_multilepton/keras-DNN ##change to your gpu account directory
#source ./setup.sh
source /cvmfs/sft.cern.ch/lcg/views/dev4cuda/latest/x86_64-centos7-gcc8-opt/setup.sh
date
time(python DNN_multi_class.py) ##change to your job file
##########################################
# Work load end

# Do not remove below this line

# list the allocated hosts
srun -l hostname
  
# list the GPU cards of the host
/usr/bin/nvidia-smi -L
echo "Allocate GPU cards : ${CUDA_VISIBLE_DEVICES}"
  
sleep 180 
