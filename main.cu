#include <cuda.h>
#include <iostream>
#include <vector>
#include <assert.h>
#include <cstdio>
#include <time.h>
#include <algorithm>

using namespace std;

//////		Vector Operations		//////

// Vector Addition

template<typename T>
__global__ void cudaAddVec(T *a, T *b, int n)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id<n)a[id]+=b[id];
}

void deviceError(string msg)
{
	cout << msg << endl;
	cudaDeviceReset();
}

template<typename T>
vector<T> vectorAdd(vector<T> a, vector<T> b)
{
	assert(a.size() == b.size());
	int n = a.size();

	T *h_a = &a[0];
	T *h_b = &b[0];

	T *d_a, *d_b;

	if(cudaMalloc(&d_a, sizeof(T) * n)!=cudaSuccess){deviceError("Error Allocating Memory To Device"); return vector<T>();};
	if(cudaMalloc(&d_b, sizeof(T) * n)!=cudaSuccess){deviceError("Error Allocating Memory To Device"); return vector<T>();};

	if(cudaMemcpy(d_a,h_a,sizeof(T) * n, cudaMemcpyHostToDevice)!=cudaSuccess){deviceError("Error Copying Variables To Device"); return vector<T>();};
	if(cudaMemcpy(d_b,h_b,sizeof(T) * n, cudaMemcpyHostToDevice)!=cudaSuccess){deviceError("Error Copying Variables To Device"); return vector<T>();};

	cudaAddVec<<<a.size()/256 + 1, 256>>>(d_a, d_b,n);

	if(cudaMemcpy(h_a, d_a, sizeof(T) * n, cudaMemcpyDeviceToHost)!=cudaSuccess){deviceError("Error Copying Variables From Device Back To Host"); return vector<T>();};

	cudaDeviceReset();

	return vector<T>(h_a, h_a+n);
}

template<typename T>
vector<T> operator+(vector<T> const &a, vector<T> const &b)
{
	return vectorAdd(a,b);
}

template <typename T>
vector<T>& operator+=(vector<T>& a, const vector<T>& b)
{
    a = a + b;
    return a;
}

// Vector Sabtraction

template<typename T>
__global__ void cudaSabVec(T *a, T *b, int n)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id<n)a[id]-=b[id];
}

template<typename T>
vector<T> vectorSab(vector<T> a, vector<T> b)
{
	assert(a.size() == b.size());
	int n = a.size();

	T *h_a = &a[0];
	T *h_b = &b[0];

	T *d_a, *d_b;

	if(cudaMalloc(&d_a, sizeof(T) * n)!=cudaSuccess){deviceError("Error Allocating Memory To Device"); return vector<T>();};
	if(cudaMalloc(&d_b, sizeof(T) * n)!=cudaSuccess){deviceError("Error Allocating Memory To Device"); return vector<T>();};

	if(cudaMemcpy(d_a,h_a,sizeof(T) * n, cudaMemcpyHostToDevice)!=cudaSuccess){deviceError("Error Copying Variables To Device"); return vector<T>();};
	if(cudaMemcpy(d_b,h_b,sizeof(T) * n, cudaMemcpyHostToDevice)!=cudaSuccess){deviceError("Error Copying Variables To Device"); return vector<T>();};

	cudaSabVec<<<a.size()/256 + 1, 256>>>(d_a, d_b,n);

	if(cudaMemcpy(h_a, d_a, sizeof(T) * n, cudaMemcpyDeviceToHost)!=cudaSuccess){deviceError("Error Copying Variables From Device Back To Host"); return vector<T>();};

	cudaDeviceReset();

	return vector<T>(h_a, h_a+n);
}

template<typename T>
vector<T> operator-(vector<T> const &a, vector<T> const &b)
{
	return vectorSab(a,b);
}

template <typename T>
vector<T>& operator-=(vector<T>& a, const vector<T>& b)
{
    a = a - b;
    return a;
}

// Vector Multiplication

template<typename T>
__global__ void cudaMultVec(T *a, T *b, int n)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id<n)a[id]*=b[id];
}

template<typename T>
__global__ void cudaMultVecScalr(T *a, T *b, int n)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id<n)a[id]*=b[0];
}

template<typename T>
vector<T> vectorMult(vector<T> a, vector<T> b)
{
	assert(a.size() == b.size());
	int n = a.size();

	T *h_a = &a[0];
	T *h_b = &b[0];

	T *d_a, *d_b;

	if(cudaMalloc(&d_a, sizeof(T) * n)!=cudaSuccess){deviceError("Error Allocating Memory To Device"); return vector<T>();};
	if(cudaMalloc(&d_b, sizeof(T) * n)!=cudaSuccess){deviceError("Error Allocating Memory To Device"); return vector<T>();};

	if(cudaMemcpy(d_a,h_a,sizeof(T) * n, cudaMemcpyHostToDevice)!=cudaSuccess){deviceError("Error Copying Variables To Device"); return vector<T>();};
	if(cudaMemcpy(d_b,h_b,sizeof(T) * n, cudaMemcpyHostToDevice)!=cudaSuccess){deviceError("Error Copying Variables To Device"); return vector<T>();};

	cudaMultVec<<<a.size()/256 + 1, 256>>>(d_a, d_b,n);

	if(cudaMemcpy(h_a, d_a, sizeof(T) * n, cudaMemcpyDeviceToHost)!=cudaSuccess){deviceError("Error Copying Variables From Device Back To Host"); return vector<T>();};

	cudaDeviceReset();

	return vector<T>(h_a, h_a+n);
}

template<typename T>
vector<T> vectorMultScalr(vector<T> a, T b)
{
	int n = a.size();

	T *h_a = &a[0];
	T *h_b = &b;

	T *d_a, *d_b;

	if(cudaMalloc(&d_a, sizeof(T) * n)!=cudaSuccess){deviceError("Error Allocating Memory To Device"); return vector<T>();};
	if(cudaMalloc(&d_b, sizeof(T) * n)!=cudaSuccess){deviceError("Error Allocating Memory To Device"); return vector<T>();};

	if(cudaMemcpy(d_a,h_a,sizeof(T) * n, cudaMemcpyHostToDevice)!=cudaSuccess){deviceError("Error Copying Variables To Device"); return vector<T>();};
	if(cudaMemcpy(d_b,h_b,sizeof(T) * n, cudaMemcpyHostToDevice)!=cudaSuccess){deviceError("Error Copying Variables To Device"); return vector<T>();};

	cudaMultVecScalr<<<a.size()/256 + 1, 256>>>(d_a, d_b,n);

	if(cudaMemcpy(h_a, d_a, sizeof(T) * n, cudaMemcpyDeviceToHost)!=cudaSuccess){deviceError("Error Copying Variables From Device Back To Host"); return vector<T>();};

	cudaDeviceReset();

	return vector<T>(h_a, h_a+n);
}

template<typename T>
vector<T> operator*(vector<T> const &a, vector<T> const &b)
{
	return vectorMult(a,b);
}

template<typename T>
vector<T> operator*(vector<T> const &a, T const &b)
{
	return vectorMultScalr(a,b);
}

template <typename T>
vector<T>& operator*=(vector<T>& a, const vector<T>& b)
{
    a = a * b;
    return a;
}

template <typename T>
vector<T>& operator*=(vector<T>& a, const T& b)
{
    a = a * b;
    return a;
}

// Vector Element Summation

template<typename T>
T sumVec(vector<T> const &a)
{
	T sum = 0;
	for(auto e : a)sum+=e;
	return sum;
}

// Vector Inner Product

template<typename T>
T dotProduct(vector<T> a, vector<T> b)
{
	vector<T> temp = a * b;
	return sumVec(temp);
}

// Vector Print

template<typename T>
void vectorPrint(vector<T> const &a)
{
	for(auto e : a)cout << e << " ";
	cout << endl;
}

template<typename T>
void operator~(vector<T> const &a)
{
	vectorPrint(a);
}

//////		Matrix Operations		//////


// Transpose

// Transpose
template<typename T>
vector< vector<T> > transpose(vector< vector<T> > const &a)
{
    vector<vector<T>> temp;
    if (a.size() > 1)
    {
    	if(true)
        {
            for (int i = 0; i < a.size(); i++)
            {
                temp.push_back(vector<T>{});
                for (int j = 0; j < a[i].size(); j++)
                {
                    temp.back().push_back(a[j][i]);
                }
            }
        }
    }
    return temp;
}


// Matrix Addition

template<typename T>
__global__ void cudaMatrixAdd(int n, T* a, T* b, T* c)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    c[row*n + col] = a[row*n + col] + b[row*n + col];
}


template<typename T>
vector< vector<T> > matrixAdd(vector< vector<T> > &va, vector< vector<T> > &vb)
{
	assert(va.size() == vb.size() && va[0].size() == vb[0].size());

    int n = va.size();

    vector<vector<T>> temp(n, vector<T>(n));

    T* a = new T[n * n];
    T* b = new T[n * n];
    T* c = new T[n * n];
    T *d_a, *d_b, *d_c;

    dim3 dimGrid(n,n, 1);
    int ts = va.size()/256 + 1;
    cudaMalloc(&d_a, n * n * sizeof(T));
    cudaMalloc(&d_b, n * n * sizeof(T));
    cudaMalloc(&d_c, n * n * sizeof(T));


    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            a[n * i + j] = va[i][j];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            b[n * i + j] = vb[i][j];

    cudaMemcpy(d_a, a, n * n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * n * sizeof(T), cudaMemcpyHostToDevice);

    cudaMatrixAdd<<<dimGrid, ts>>>(n, d_a, d_b, d_c);

    cudaMemcpy(c, d_c, n * n * sizeof(T), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            temp[i][j] = c[n * i + j];

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return temp;
}


// Matrix Sabtraction
template<typename T>
__global__ void cudaMatrixSab(int n, T* a, T* b, T* c)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    c[row*n + col] = a[row*n + col] - b[row*n + col];
}


template<typename T>
vector< vector<T> > matrixSab(vector< vector<T> > &va, vector< vector<T> > &vb)
{
	assert(va.size() == vb.size() && va[0].size() == vb[0].size());

    int n = va.size();

    vector<vector<T>> temp(n, vector<T>(n));

    T* a = new T[n * n];
    T* b = new T[n * n];
    T* c = new T[n * n];
    T *d_a, *d_b, *d_c;

    dim3 dimGrid(n,n, 1);
    int ts = va.size()/256 + 1;
    cudaMalloc(&d_a, n * n * sizeof(T));
    cudaMalloc(&d_b, n * n * sizeof(T));
    cudaMalloc(&d_c, n * n * sizeof(T));


    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            a[n * i + j] = va[i][j];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            b[n * i + j] = vb[i][j];

    cudaMemcpy(d_a, a, n * n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * n * sizeof(T), cudaMemcpyHostToDevice);

    cudaMatrixSab<<<dimGrid, ts>>>(n, d_a, d_b, d_c);

    cudaMemcpy(c, d_c, n * n * sizeof(T), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            temp[i][j] = c[n * i + j];

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return temp;
}

// Matrix Multiplication

template<typename T>
__global__ void cudaMatrixMult(int m, int n, int k, T* a, T* b, T* c)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < m && col < k)
    {
        T tmp = (T)0;
        for (int i = 0; i < n; i++)
            tmp += a[row * n + i] * b[col + i * k];
        c[row * k + col] = tmp;
    }
}


template<typename T>
vector< vector<T> > matrixMult(vector< vector<T> > const& va, vector< vector<T> > const& vb)
{
    int m = va.size();
    int n = va[0].size();
    int k = vb[0].size();

    vector<vector<T>> temp(m, vector<T>(k));

    T* a = new T[m * n];
    T* b = new T[n * k];
    T* c = new T[m * k];
    T *d_a, *d_b, *d_c;

    dim3 dimGrid(k,m, 1);
    int ts = va.size()/256 + 1;
    cudaMalloc(&d_a, m * n * sizeof(T));
    cudaMalloc(&d_b, n * k * sizeof(T));
    cudaMalloc(&d_c, m * k * sizeof(T));


    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            a[n * i + j] = va[i][j];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < k; j++)
            b[k * i + j] = vb[i][j];

    cudaMemcpy(d_a, a, m * n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * k * sizeof(T), cudaMemcpyHostToDevice);

    cudaMatrixMult<<<dimGrid, ts>>>(m, n, k, d_a, d_b, d_c);

    cudaMemcpy(c, d_c, m * k * sizeof(T), cudaMemcpyDeviceToHost);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++)
            temp[i][j] = c[k * i + j];

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return temp;
}

// Matrix Scalar Mult
template<typename T>
__global__ void cudaMatrixScalarMult(int n, T *a, T *b, T *c)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    c[row*n + col] = a[row*n + col] * b[0];
}


template<typename T>
vector< vector<T> > matrixScalarMult(vector< vector<T> > va, T vb)
{
    int n = va.size();

    vector<vector<T>> temp(n, vector<T>(n));

    T* a = new T[n * n];
    T* b = &vb;
    T* c = new T[n * n];
    T *d_a, *d_b, *d_c;

    dim3 dimGrid(n,n, 1);
    int ts = va.size()/256 + 1;
    cudaMalloc(&d_a, n * n * sizeof(T));
    cudaMalloc(&d_b, sizeof(T));
    cudaMalloc(&d_c, n * n * sizeof(T));


    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            a[n * i + j] = va[i][j];

    cudaMemcpy(d_a, a, n * n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(T), cudaMemcpyHostToDevice);

    cudaMatrixScalarMult<<<dimGrid, ts>>>(n, d_a, d_b, d_c);

    cudaMemcpy(c, d_c, n * n * sizeof(T), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            temp[i][j] = c[n * i + j];

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return temp;
}
// Matrix Determinant

template<typename T>
vector< vector<T> > shave(vector< vector<T> > a, int i)
{
    a.erase(a.begin() + 0);
    for (int j = 0; j < a.size(); j++)
    {
        a[j].erase(begin(a[j]) + i);
    }

    return a;
}

template<typename T>
vector< vector<T> > shave(vector< vector<T> > a, int i, int j)
{
    a.erase(a.begin() + j);
    for (int j = 0; j < a.size(); j++)
    {
        a[j].erase(begin(a[j]) + i);
    }

    return a;
}

template<typename T>
float det2x2(vector< vector<T> > const &a)
{
    return a[0][0] * a[1][1] - a[0][1] * a[1][0];
}

template<typename T>
float detNxN(vector< vector<T> > const &a);

template<typename T>
float det(vector< vector<T> > const &a)
{
    assert(a.size() == a[0].size());
    if (a[0].size() == 2 && a.size() == 2)return det2x2(a);
    else return detNxN(a);
}

template<typename T>
float detNxN(vector< vector<T> > const &a)
{
    float sum = 0;
    for (int i = 0; i < a.size(); i++)
    {
        if ((i + 1) % 2 == 0)
        {
            sum += a[0][i] * det(shave(a, i)) * (-1);
        }
        else
        {
            sum += a[0][i] * det(shave(a, i)) * (1);
        }
    }
    return sum;
}

// Cofactor Matrix
template<typename T>
vector< vector<T> > cof(vector< vector<T> > const &a)
{
    vector< vector<T> > cofactors;
    for (int i = 0; i < a[0].size(); i++)
    {
        cofactors.push_back(vector<T>{});
        for (int j = 0; j < a.size(); j++)
        {
            int g = ((i + 1 + j) % 2 == 0) ? -1 : 1;
            cofactors.back().push_back(det(shave(a, i, j)) * g);
        }
    }

    cofactors = transpose(cofactors);
    return cofactors;
}

// Matrix Inverse

template<typename T>
vector< vector<T> > inv(vector< vector<T> > a)
{
    if (a[0].size() >= 3)
    {
        float detr = det(a);
        vector< vector<T> > inv = cof(a);
        inv = transpose(inv);
        inv = matrixScalarMult(inv,1/detr);
        return inv;
    }
    else
    {
        vector< vector<T> > inv({ { a[1][1],a[0][1] * -1 },{ a[1][0] * -1, a[0][0] } });
		matrixScalarMult(inv, (1/det(a)));
		return inv;
    }
}

// Printing Matrices

template<typename T>		
void printMatrix(vector< vector<T> > const &c)
{
	for(int i = 0; i < c.size(); i++)
    {
    	for(int j = 0; j < c[i].size(); j++)
    	{
    		cout << c[i][j] << "\t";
    	}
    	cout << endl;
    }
}

int main()
{
	//////		Testing		//////

	vector< vector<float> > a;
    vector< vector<float> > b;
    for(int i = 1; i <= 4; i++){
    	a.push_back(vector<float>());
    	b.push_back(vector<float>());
    	for(int j = 1; j <= 4; j++){
    		a.back().push_back(rand()%10);
    		b.back().push_back(rand()%10);
    	}
    }	

    vector< vector<float> > test = {{5,7,8},{0,5,4},{6,7,9}};
    vector< vector<float> > iTest = inv(test);
    printMatrix(iTest);
    cout << endl;
    cout << "Gaze upon the power of the GPU!" << endl;
    vector < vector<float> > ss{ { 8,3 },{ 1,2 } };
    vector < vector<float> > result{ {46},{9} };
    float detr = det(ss);
    ss = inv(ss);
    matrixMult(ss,result);
    printMatrix(ss);
    cout << endl;
	vector< vector<float> > c = matrixAdd(a,b);
    printMatrix(a);cout << '\n';
    printMatrix(b);cout << '\n';
    printMatrix(c);
    
    //////		(cuda)Success!!		////// \(^o^)/
	return 0;
}
