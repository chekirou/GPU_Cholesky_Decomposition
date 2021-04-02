# Execution 

`Execution.ipynb` contient le script d'execution ainsi que celui de l'affichage des graphes et de l'architecture utilisée.
Il a été executé sur google collaboratory.

Ou executer en faisant :

!nvcc -o cholesky cholesky.cu -gencode arch=compute_35,code=compute_35

! ./cholesky 

# functions: 
- LDLt_max_k : version collaborative sur les lignes 
- LDLt_max_col_k : version collaborative sur les colonnes 
- LDLt_k : version avec un thread par system
- wraper: execute un des kernel selon le mode (0: LDLt_max_k, 1: LDLt_max_col_k, 2 : LDLt_k )
- comparison : execute les 3 kernels et stockes les temps dans le fichier results.txt
- Dans le main : execution pour 32 * 16384 matrices de taille 16*16.

# GPU utilisé 

name = b'Tesla T4'
maxThreadsPerBlock = 1024
maxBlockDimX = 1024
maxBlockDimY = 1024
maxBlockDimZ = 64
maxGridDimX = 2147483647
maxGridDimY = 65535
maxGridDimZ = 65535
maxSharedMemoryPerBlock = 49152
asyncEngineCount = 3
canMapHostMemory = 1
multiProcessorCount = 40
warpSize = 32
unifiedAddressing = 1
pciBusID = 0
pciDeviceID = 4
