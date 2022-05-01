#!/bin/bash -l
#SBATCH --partition=medium        ///// Partición del clúster según el tiempo de duración.
#SBATCH --job-name=wfield         ///// Nombre de la tarea en slurmtop (comando en bash para mostrarte todos los usuarios) y sjobstat (muestra tus tareas en curso y finalizadas)
#SBATCH --ntasks=1                ///// Número de tareas, repeticiones del mismo código.
#SBATCH --cpus-per-task=1         ///// Número de núcleos. Por generalmente uno para que no te haga esperar.
#SBATCH --mem-per-cpu=1G          ///// Memoria requerida
#SBATCH --time=05:40:00           ///// Tiempo de duración máxima.

#SBATCH --output BloqueHopp00.out ///// Nombre del archivo que contiene la impresión a consola ( print(blablabla) )

module load python
python wfield.py         

cp -r . /data/finite/carlosbe
cd

rm -rf $scratch
unset scratch

exit 0

