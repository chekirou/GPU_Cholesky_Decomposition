// author : Hakim CHEKIROU


#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// Function that catches the error 
void testCUDA(cudaError_t error, const char *file, int line)  {

	if (error != cudaSuccess) {
	   printf("There is an error : %s in file %s at line %d\n",cudaGetErrorString(error), file, line);
       exit(EXIT_FAILURE);
	} 
    
}

// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

//========================================
// version collaborative sur les lignes 
//========================================
__global__ void LDLt_max_k(float *a, float *y, int n){
    
    // shared memory
    extern __shared__ float sA[];
    //Local integers
    int i, k, n2, n2p1, nt;
    int tidx = threadIdx.x%n; // indice dans le groupe de thread
    int Qt = (threadIdx.x-tidx)/n; // indice du system dans le block
    int gbx = Qt + blockIdx.x*(blockDim.x/n); // indice du system dans tout les system
    
    n2 = (n*n+n)/2; // taille de la moitié de la matrice
    n2p1=n2+n; // taille de tout un system
    nt = Qt*n2p1; // indice dans la memoire du system 
    // chargement des données
    // seulement la matrice triangulaire superieure
    for(i=0; i<n; i++)
    {
        if(tidx >= i){
            // on respecte l'indexation dans le code fourni
            // depuis la fin on revient en arriere de n-i lignes pour acceder à la ligne i
            // on rajoute -i car on stocke seulement la moitié 
            sA[nt+n2-(n-i)*((n-i)+1)/2+tidx-i] = a[gbx*n*n+i*n+tidx];
        }
    }
    // chaque thread charge une valeur 
    sA[nt+n2+tidx] = y[gbx*n+tidx]; 

	__syncthreads();
    // LDLt factorization 
    // LDLt X = Y
    for(i=n; i>0; i--){
        if(tidx==0){
            for(k=n; k>i; k--){
                sA[nt+n2-i*(i+1)/2] -= sA[nt+n2-k*(k+1)/2]*
                sA[nt+n2-k*(k+1)/2+k-i]*
                sA[nt+n2-k*(k+1)/2+k-i];
            }
        }
        __syncthreads();
        if(tidx<i-1){
            sA[nt+n2-i*(i+1)/2+tidx+1] /= sA[nt+n2-i*(i+1)/2];
            for(k=n; k>i; k--){
                sA[nt+n2-i*(i+1)/2+tidx+1] -= sA[nt+n2-k*(k+1)/2]*
                sA[nt+n2-k*(k+1)/2+k-i]*
                sA[nt+n2-k*(k+1)/2+tidx+1+k-i]/
                sA[nt+n2-i*(i+1)/2];
            }
        }
        __syncthreads();
    }

    // resolution de L Z = Y
    // premiere valeur deja calculé
    for(k=0; k<n-1; k++){
        // k va de 0 a tidx-1 inclus
        if(tidx>k){
            // pour les colonnes dont le z deja calculé
            // z_tidk = y_tidk - sum L_tidk_k * z_k
            // dans ce cas on inverse car on a calculé l_t
            sA[nt+n2+tidx] -= sA[nt+n2-(n-k)*((n-k)+1)/2+tidx-k]*sA[nt+n2+k];
        }
        __syncthreads();
    }
    // y_tidx /= d_tidx_tidx
    sA[nt+n2+tidx] /= sA[nt+n2-(n-tidx)*(n-tidx+1)/2];
    __syncthreads();
    // resolution de L_t X = Y
    for(k=n-1; k>0; k--){
        // k va de tidx+1 a n-1
        if(tidx<k){
            // x_tidx = y_tidx - sum L_k_tidx * x_k
            // on inverse aussi  car on a L_t pas L
            sA[nt+n2+tidx] -= sA[nt+n2-(n-tidx)*(n-tidx+1)/2+k-tidx]*
							  sA[nt+n2+k];
        }
        __syncthreads();
    }

	// copie du resultat
	y[gbx*n+tidx] = sA[nt+n2+tidx];

}



//=============================================
// version sur colonne
//=============================================


__global__ void LDLt_max_col_k(float *a, float *y, int n){
    
    // shared memory
    extern __shared__ float sA[];
    //Local integers
    int i, k, n2, n2p1, nt;
    int tidx = threadIdx.x%n; // indice dans le groupe de thread
    int Qt = (threadIdx.x-tidx)/n; // indice du systeme dans le block
    int gbx = Qt + blockIdx.x*(blockDim.x/n); // indice du system dans tout les system
    
    n2 = (n*n+n)/2; 
    n2p1=n2+n;
    nt = Qt*n2p1;
    // chargement de la partie inferieure
    for(i=0; i<n; i++)
    {
        if(tidx <= i){
            // indexation simple
            sA[nt+i*(i+1)/2+tidx] = a[gbx*n*n+i*n+tidx];
        }
    }
    sA[nt+n2+tidx] = y[gbx*n+tidx]; 

	__syncthreads();
    
    for(i = 0; i< n; i++){
        // thread 0 se charge de Dii
        if(tidx==0){
            for(k=0; k<i; k++){
                sA[nt+i*(i+1)/2+i] -= sA[nt+i*(i+1)/2+k]*
                                                sA[nt+i*(i+1)/2+k]*
                                                sA[nt+k*(k+1)/2+k];
            }
        }
        __syncthreads();
        // chaque thread s'occupe d'une ligne tidx dans la colonne i
        if(tidx>i){
        
            // division by D_i_i
            sA[nt+tidx*(tidx+1)/2+i] /= sA[nt+i*(i+1)/2+i];

            for (k=0; k<i; k++){
                sA[nt+tidx*(tidx+1)/2+i] -= sA[nt+tidx*(tidx+1)/2+k]*
                                                    sA[nt+i*(i+1)/2+k]*
                                                    sA[nt+k*(k+1)/2+k]/
                                                    sA[nt+i*(i+1)/2+i];
            }
        
        }
        __syncthreads();
    }
    // resolution de L Z = Y
    for(k=0; k<n-1; k++){
        // pour les colonnes dont le z deja calculé
        if(tidx>k){
            
            // z_tidk = y_tidk - sum L_tidk_k * z_k
            sA[nt+n2+tidx] -= sA[nt+tidx*(tidx+1)/2+k]*sA[nt+n2+k];
        }
        __syncthreads();
    }
    // y_tidx /= d_tidx_tidx
    sA[nt+n2+tidx] /= sA[nt+tidx*(tidx+1)/2+tidx];
    __syncthreads();
    for(k=n-1; k>0; k--){
        // x_tidx = y_tidx - sum L_k_tidx * x_k
        if(tidx<k){
            sA[nt+n2+tidx] -= sA[nt+k*(k+1)/2+tidx]*
							  sA[nt+n2+k];
        }
        __syncthreads();
    }

    // chargement de la solution en global
	y[gbx*n+tidx] = sA[nt+n2+tidx];
}


//=======================================
// version avec un seul thread par system
//=======================================


__global__ void LDLt_k(float *a, float *y, int n){
    
    //thread indentifier in the grid
    int tidx = threadIdx.x + blockIdx.x*blockDim.x;
    // shared memory
    extern __shared__ float sA[];
    //Local integers
    int i, j, k, n2, n2p1;
    
    n2 = (n*n+n)/2; 
    n2p1=n2+n;

    //copy the lower 
    for(i=0; i<n; i++){
        for(j=0; j<=i; j++){
            sA[threadIdx.x*n2p1+i*(i+1)/2+j] = a[tidx*n*n+i*n+j];
        }
    }
    // copy the value
    for(i=0; i<n; i++){
        sA[threadIdx.x*n2p1+n2+i] = y[tidx*n+i];
    }
    // perform the LDLt decomposition
    for(i = 0; i< n; i++){
        for(j=0; j<i; j++){
            // division by D_j_j
            sA[threadIdx.x*n2p1+i*(i+1)/2+j] /= sA[threadIdx.x*n2p1+j*(j+1)/2+j];
            for (k=0; k<j; k++){
                sA[threadIdx.x*n2p1+i*(i+1)/2+j] -= sA[threadIdx.x*n2p1+i*(i+1)/2+k]*
                                                    sA[threadIdx.x*n2p1+j*(j+1)/2+k]*
                                                    sA[threadIdx.x*n2p1+k*(k+1)/2+k]/
                                                    sA[threadIdx.x*n2p1+j*(j+1)/2+j];
            }
        }
        for(k=0; k<i; k++){
            sA[threadIdx.x*n2p1+i*(i+1)/2+i] -= sA[threadIdx.x*n2p1+i*(i+1)/2+k]*
                                                sA[threadIdx.x*n2p1+i*(i+1)/2+k]*
                                                sA[threadIdx.x*n2p1+k*(k+1)/2+k];
        }

    }
    // resolving the system
    // solving LZ=Y 
    for(i=0; i< n; i++){
        for(k=0; k<i; k++){
            sA[threadIdx.x*n2p1+n2+i] -=    sA[threadIdx.x*n2p1+i*(i+1)/2+k]*
                                            sA[threadIdx.x*n2p1+n2+k];
        }
    }
    for(i=n-1; i>=0; i--){
        sA[threadIdx.x*n2p1+n2+i] /= sA[threadIdx.x*n2p1+i*(i+1)/2+i];
        for (k=i+1; k<n; k++){
            sA[threadIdx.x*n2p1+n2+i]-= sA[threadIdx.x*n2p1+k*(k+1)/2+i] *
                                        sA[threadIdx.x*n2p1+n2+k];

        }
    }
    // chargement de la solution
    for(i=0;i<n; i++){
        y[tidx*n+i] = sA[threadIdx.x*n2p1  + n2 + i];
    }
    

}



//==================================
//wrapers
//==================================

float wraper(int Dim, int minTB,int NB, int mode){
    int count;
	cudaDeviceProp prop;

    int i, j, k;
    
    // the number of matrices to invert
    int size = NB * minTB;
    // parameter to fill the matrix with
    float rho;
    // the matrix and the value vector
    float *A, *AGPU, *Y, *YGPU;
    testCUDA(cudaGetDeviceCount(&count));
	testCUDA(cudaGetDeviceProperties(&prop, count-1));

    float TimerAddOne;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //memory allocation
    A = (float *)calloc(size*Dim*Dim, sizeof(float));
    Y = (float *)calloc(size*Dim, sizeof(float));
    // GPU mem allocation 
    cudaMalloc(&AGPU, size*Dim*Dim*sizeof(float));
    cudaMalloc(&YGPU, size*Dim*sizeof(float));

    // initialization 
    for(i=0; i< size; i++){
        rho = 1.0f/(1.1f+i);
        for(j=0; j<Dim; j++){
            for(k=0; k<Dim; k++){
                if(j==k){
                    A[i*Dim*Dim+j*Dim+k] = 1.0f;
                }
                else{
                    A[i*Dim*Dim+j*Dim+k] = rho;
                    
                }
                
            }
            Y[j+i*Dim] = 0.5f*j;
        }
    }
    
    // transfer matrices to GPU
    cudaMemcpy(AGPU, A, size*Dim*Dim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(YGPU, Y, size*Dim*sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(start, 0);
   
    // resolving the system AX=Y using choleky decompostition 
    switch (mode) {
    case 0: // version avec lignes
    LDLt_max_k<<<NB , Dim*minTB, minTB * ((Dim*Dim+Dim)/2+Dim)*sizeof(float)>>>(AGPU, YGPU, Dim);
    break;
    case 1: // version colonnes
    LDLt_max_col_k<<<NB , Dim*minTB, minTB * ((Dim*Dim+Dim)/2+Dim)*sizeof(float)>>>(AGPU, YGPU, Dim);
    break;
    case 2: // version avec une seul thread par systeme
    LDLt_k<<<NB , minTB, minTB * ((Dim*Dim+Dim)/2+Dim)*sizeof(float)>>>(AGPU, YGPU, Dim);
    break;
    }
    
    
    testCUDA(cudaEventRecord(stop, 0));
    testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimerAddOne, start, stop));
    testCUDA(cudaMemcpy(Y, YGPU, size*Dim*sizeof(float), cudaMemcpyDeviceToHost));

    i=79;
    printf("solution for system : %d \n", i);
    
    printf("[");
    for(j=0; j<Dim; j++){
        
        printf("%f, ", Y[j+i*Dim]);
    }
    printf("]\n");

    printf("GPU Timer: %f ms\n", TimerAddOne);
    // free memory
    free(A);
    cudaFree(AGPU);
    free(Y);
    cudaFree(YGPU);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return TimerAddOne;
}


// Fonction pour comparer les 3 versions
// et enregistre les resultats dans un fichier

void comparison(){
    FILE *fptr;
    fptr = fopen("./results.txt","w");
    // mis a 4 car pour dim =64, la memoire shared atteint sa limite
    // mis a 1 pour aller jusqu'a 155 (taille max en memoire shared)
    int minTB = 4; 
    int NB = 16384; // * 4 si on reduit minTB a 1
    if(fptr == NULL)
    {
        printf("Error!");   
        exit(1);             
    }
    fprintf(fptr, "Dimension; monoThread;Row; Col\n");
    float col, mono, row;
    for(int d = 4; d < 64; d++){
        printf("==================================== \n");
        printf("dimension %d :\n", d);
        printf("==================================== \n");
        printf("------------------------------------ \n");
        printf("Mono thread \n");

        mono = wraper(d, minTB, NB, 2);

        printf("------------------------------------ \n");
        printf("Row version \n");

        row = wraper(d, minTB, NB, 0);

        printf("------------------------------------ \n");
        printf("Col version \n");

        col = wraper(d, minTB, NB, 1);

        fprintf(fptr,"%d ; %f; %f ; %f \n",d,mono, row, col);
    }
    fclose(fptr);
}
int main(){
    int minTB = 32;
    int NB = 16384;
    int d = 16;
   
    printf("==================================== \n");
    printf("dimension %d :\n", d);
    printf("==================================== \n");
    printf("------------------------------------ \n");
    printf("Mono thread \n");

    wraper(d, minTB, NB, 2);

    printf("------------------------------------ \n");
    printf("Row version \n");

    wraper(d, minTB, NB, 0);

    printf("------------------------------------ \n");
    printf("Col version \n");

    wraper(d, minTB, NB, 1);

    
    return 0;
}
/*
int main(){
    comparison();
    return 0;
}
*/