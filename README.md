# DAE

cd WIE     # get into the WIE folder

pc-01$ python t2.py --job_name="ps" --task_index=0 

pc-02$ python t2.py --job_name="worker" --task_index=0 

pc-03$ python t2.py --job_name="worker" --task_index=1 

pc-04$ python t2.py --job_name="worker" --task_index=2 
