#!/bin/bash

# retrieving storyline data from tape archive using SLK
# example file, change according to needs
# var151 (MSLP) monthly files
# do not make the assignment to big, it is better to submit several smaller codes to parallel the retrieval (hopefully that will make it faster)

# module load slk
# slk login (use gnumber 260212 and levante password)
# Send this script from pc to server <scp -r D:\paper_3\code\SLKretrieve_tape_2011.sh g260212@levante.dkrz.de:/work/gg0301/from_Mistral/gg0301/g260212/>

#SBATCH --job-name=2011_retrieval   					 # Specify job name CHNAGE
#SBATCH --output=2011_retrieval.%j                # name for standard output log file, same as job-name
#SBATCH --error=2011_retrieval.%j                 # name for standard error output log file, it is better it is the same as ouput
#SBATCH --partition=shared                                # Specify partition name, do not change
#SBATCH --ntasks=1                                        # Specify max. number of tasks to be invoked, do not change
#SBATCH --mem=6GB                                         # allocated memory for the script, do not change
#SBATCH --time=48:00:00                                   # Set a limit on the total run time, 48 hours for a retrieval job of 12 months of data is OK at this moment in time
#SBATCH --account=gg0301                                  # our project code, do not change   

# INPUT
tape=/arch/gg0301/g260132/ECHAM6_T255l95_sn_SU_1015/
exp=BOT_t_HR255_Nd_SU_1015

# Output
storm=("2011")
vars=("151" "165" "166") #put in var number of interest, 151 is MSLP.
goal=/work/gg0301/from_Mistral/gg0301/g260212/tape_retrieval/${storm} #change to the path on Levante you want to place the data
mkdir -p ${goal}

months=("08" "09" "10")

for var in ${vars[@]} ; do
    for year in 2011 ; do
        for mon in ${months[@]} ; do
            for mem in counter factual plus2 ; do
                for num in 1 2 3 ; do

                    slk retrieve ${tape}/${mem}_${num}/${exp}_${mem}_${num}_${year}${mon}.${var}.nc ${goal}/${exp}_${mem}_${num}_${year}${mon}.${var}.nc

                    if [ $? -ne 0 ] ; then
                        >&2 echo "an error occurred in slk retrieve call"
                    else
                        echo "retrieval successful"	    

                    fi
            
                done
            done
        done
    done
done


exit
