'''
Author : Karan Chauhan
github : @Karan-Chauhan19
Organization : L.J University
'''

import torch 
import subprocess


class GPU_Config:

    def __init__(self):
        pass
    def check_gpu_configration() -> None:
        """
        This function checks the GPU configuration of the system.

        params
        - (None)

        return
        - (None)
        """

        if torch.cuda.is_available(): # This function check GPU is avilable or not if present then it return True else False
           
           num_gpu = torch.cuda.device_count() # get the number of GPU
           print(f"Total number of GPU's available: {num_gpu}") 

           for i in range(num_gpu):
               gpu_name = torch.cuda.get_device_name() # get the name of the GPU
               print("- GPU name : {}".format(gpu_name))
            
           command = 'nvidia - smi' # set the command
           result = subprocess.run(command, shell=True, capture_output=True, text=True)

           if result.returncode == 0:
               print(result.stdout)
           else:
               print("- Error massage : {}".format(result.stderr))
        else :
            print("No GPU's available")
            print("Training start on the CPU ! ")