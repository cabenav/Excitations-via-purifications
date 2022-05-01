#!/bin/bash
#SBATCH --partition=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=wfield                   # Job name
#SBATCH --mail-type=END,FAIL                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=cabenavir@gmail.com     # Where to send mail	
#SBATCH --ntasks=1                          # Run on a single CPU
#SBATCH --mem-per-cpu=1G                    # Job memory request
#SBATCH --time=05:00:00                     # Time limit hrs:min:sec
#SBATCH --output=BloqueHopp00.out          # Standard output and error log
set -e

module load python  

mkdir /data/finite/carlosbe/results
cd /data/finite/carlosbe/results

cp /data/finite/carlosbe/wfieldrun.py
python wfieldrun.py      

cp -r results /data/finite/carlosbe

exit 0
