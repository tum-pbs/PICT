#pragma once

#ifndef _INCLUDE_TRANSFORMATIONS
#define _INCLUDE_TRANSFORMATIONS

#include "custom_types.h"
#include <cmath>


#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

template <typename scalar_t, index_t SIZE>
union Vector{
	scalar_t a[SIZE]; //row, col
};

template <typename scalar_t>
union Vector<scalar_t, 1>{
    struct{
		scalar_t x;
    };
	scalar_t a[1]; //row, col
};

template <typename scalar_t>
union Vector<scalar_t, 2>{
    struct{
		scalar_t x;
		scalar_t y;
    };
	scalar_t a[2]; //row, col
};

template <typename scalar_t>
union Vector<scalar_t, 3>{
    struct{
		scalar_t x;
		scalar_t y;
		scalar_t z;
    };
	scalar_t a[3]; //row, col
};

template <typename scalar_t>
union Vector<scalar_t, 4>{
    struct{
		scalar_t x;
		scalar_t y;
		scalar_t z;
		scalar_t w;
    };
	scalar_t a[4]; //row, col
};

template <typename scalar_t, index_t LENGTH>
HOST_DEVICE Vector<scalar_t, LENGTH - 1> drop_element(Vector<scalar_t, LENGTH> v, index_t drop_index){
    Vector<scalar_t, LENGTH - 1> ret;
    index_t retIdx = 0;
    for(index_t vIdx = 0; vIdx < LENGTH; vIdx++){
        if(vIdx != drop_index){
            ret.a[retIdx] = v.a[vIdx];
            retIdx++;
        }
    }
    return ret;
}


template <typename scalar_t, index_t SIZE>
union MatrixSquare{
    scalar_t a[SIZE][SIZE]; //row, col
    Vector<scalar_t, SIZE> v[SIZE]; //vectors are rows
};


template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE Vector<scalar_t, SIZE> operator+(Vector<scalar_t, SIZE> a, const Vector<scalar_t, SIZE> &b){
    for(index_t i=0; i<SIZE; ++i){
        a.a[i] += b.a[i];
    }
    return a;
}
template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE Vector<scalar_t, SIZE> operator+(Vector<scalar_t, SIZE> a, const scalar_t &s){
    for(index_t i=0; i<SIZE; ++i){
        a.a[i] += s;
    }
    return a;
}
template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE Vector<scalar_t, SIZE> operator+(const scalar_t &s, Vector<scalar_t, SIZE> a){
    for(index_t i=0; i<SIZE; ++i){
        a.a[i] += s;
    }
    return a;
}

template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE Vector<scalar_t, SIZE>& operator+=(Vector<scalar_t, SIZE> &a, const Vector<scalar_t, SIZE> &b){
    for(index_t i=0; i<SIZE; ++i){
        a.a[i] += b.a[i];
    }
    return a;
}
template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE Vector<scalar_t, SIZE>& operator+=(Vector<scalar_t, SIZE> &a, const scalar_t &s){
    for(index_t i=0; i<SIZE; ++i){
        a.a[i] += s;
    }
    return a;
}

template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE Vector<scalar_t, SIZE> operator-(Vector<scalar_t, SIZE> a, const Vector<scalar_t, SIZE> &b){
    for(index_t i=0; i<SIZE; ++i){
        a.a[i] -= b.a[i];
    }
    return a;
}
template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE Vector<scalar_t, SIZE> operator-(Vector<scalar_t, SIZE> a, const scalar_t &s){
    for(index_t i=0; i<SIZE; ++i){
        a.a[i] -= s;
    }
    return a;
}
template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE Vector<scalar_t, SIZE> operator-(const scalar_t &s, Vector<scalar_t, SIZE> a){
    for(index_t i=0; i<SIZE; ++i){
        a.a[i] = s - a.a[i];
    }
    return a;
}

template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE Vector<scalar_t, SIZE>& operator-=(Vector<scalar_t, SIZE> &a, const Vector<scalar_t, SIZE> &b){
    for(index_t i=0; i<SIZE; ++i){
        a.a[i] -= b.a[i];
    }
    return a;
}
template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE Vector<scalar_t, SIZE>& operator-=(Vector<scalar_t, SIZE> &a, const scalar_t &s){
    for(index_t i=0; i<SIZE; ++i){
        a.a[i] -= s;
    }
    return a;
}

template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE Vector<scalar_t, SIZE> operator*(Vector<scalar_t, SIZE> a, const Vector<scalar_t, SIZE> &b){
    for(index_t i=0; i<SIZE; ++i){
        a.a[i] *= b.a[i];
    }
    return a;
}
template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE Vector<scalar_t, SIZE> operator*(Vector<scalar_t, SIZE> a, const scalar_t &s){
    for(index_t i=0; i<SIZE; ++i){
        a.a[i] *= s;
    }
    return a;
}
template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE Vector<scalar_t, SIZE> operator*(const scalar_t &s, Vector<scalar_t, SIZE> a){
    for(index_t i=0; i<SIZE; ++i){
        a.a[i] *= s;
    }
    return a;
}

template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE Vector<scalar_t, SIZE>& operator*=(Vector<scalar_t, SIZE> &a, const Vector<scalar_t, SIZE> &b){
    for(index_t i=0; i<SIZE; ++i){
        a.a[i] *= b.a[i];
    }
    return a;
}
template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE Vector<scalar_t, SIZE>& operator*=(Vector<scalar_t, SIZE> &a, const scalar_t &s){
    for(index_t i=0; i<SIZE; ++i){
        a.a[i] *= s;
    }
    return a;
}

template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE Vector<scalar_t, SIZE> operator<(Vector<scalar_t, SIZE> a, const Vector<scalar_t, SIZE> &b){
    for(index_t i=0; i<SIZE; ++i){
        a.a[i] = a.a[i] < b.a[i];
    }
    return a;
}
template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE Vector<scalar_t, SIZE> operator<(Vector<scalar_t, SIZE> a, const scalar_t &s){
    for(index_t i=0; i<SIZE; ++i){
        a.a[i] = a.a[i] < s;
    }
    return a;
}
template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE Vector<scalar_t, SIZE> operator<(const scalar_t &s, Vector<scalar_t, SIZE> a){
    for(index_t i=0; i<SIZE; ++i){
        a.a[i] = s < a.a[i];
    }
    return a;
}

template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE Vector<scalar_t, SIZE> operator<=(Vector<scalar_t, SIZE> a, const Vector<scalar_t, SIZE> &b){
    for(index_t i=0; i<SIZE; ++i){
        a.a[i] = a.a[i] <= b.a[i];
    }
    return a;
}
template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE Vector<scalar_t, SIZE> operator<=(Vector<scalar_t, SIZE> a, const scalar_t &s){
    for(index_t i=0; i<SIZE; ++i){
        a.a[i] = a.a[i] <= s;
    }
    return a;
}
template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE Vector<scalar_t, SIZE> operator<=(const scalar_t &s, Vector<scalar_t, SIZE> a){
    for(index_t i=0; i<SIZE; ++i){
        a.a[i] = s <= a.a[i];
    }
    return a;
}

template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE Vector<scalar_t, SIZE> operator>(Vector<scalar_t, SIZE> a, const Vector<scalar_t, SIZE> &b){
    for(index_t i=0; i<SIZE; ++i){
        a.a[i] = a.a[i] > b.a[i];
    }
    return a;
}
template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE Vector<scalar_t, SIZE> operator>(Vector<scalar_t, SIZE> a, const scalar_t &s){
    for(index_t i=0; i<SIZE; ++i){
        a.a[i] = a.a[i] > s;
    }
    return a;
}
template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE Vector<scalar_t, SIZE> operator>(const scalar_t &s, Vector<scalar_t, SIZE> a){
    for(index_t i=0; i<SIZE; ++i){
        a.a[i] = s > a.a[i];
    }
    return a;
}

template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE Vector<scalar_t, SIZE> operator>=(Vector<scalar_t, SIZE> a, const Vector<scalar_t, SIZE> &b){
    for(index_t i=0; i<SIZE; ++i){
        a.a[i] = a.a[i] >= b.a[i];
    }
    return a;
}
template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE Vector<scalar_t, SIZE> operator>=(Vector<scalar_t, SIZE> a, const scalar_t &s){
    for(index_t i=0; i<SIZE; ++i){
        a.a[i] = a.a[i] >= s;
    }
    return a;
}
template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE Vector<scalar_t, SIZE> operator>=(const scalar_t &s, Vector<scalar_t, SIZE> a){
    for(index_t i=0; i<SIZE; ++i){
        a.a[i] = s >= a.a[i];
    }
    return a;
}

template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE bool operator==(Vector<scalar_t, SIZE> a, const Vector<scalar_t, SIZE> &b){
    for(index_t i=0; i<SIZE; ++i){
        if(a.a[i] != b.a[i]){ return false; }
    }
    return true;
}

template<typename scalar_t, index_t SIZE>
inline HOST_DEVICE scalar_t norm(Vector<scalar_t, SIZE> v){
    scalar_t acc = 0;
    for(index_t i = 0; i < SIZE; i++){
        acc += v.a[i] * v.a[i]; 
    }
    return std::sqrt(acc);
}

template<typename scalar_t, index_t SIZE>
inline HOST_DEVICE Vector<scalar_t, SIZE> abs(Vector<scalar_t, SIZE> v){
    for(index_t i = 0; i < SIZE; i++){
        v.a[i] = abs(v.a[i]); 
    }
    return v;
}

template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE scalar_t dot(const Vector<scalar_t, SIZE> &a, const Vector<scalar_t, SIZE> &b){
    scalar_t result=0;
    for(index_t i=0; i<SIZE; ++i){
        result += a.a[i]*b.a[i];
    }
    return result;
}


template <typename scalar_t>
inline HOST_DEVICE scalar_t cross(const Vector<scalar_t, 2> &a, const Vector<scalar_t, 2> &b){
    return a.a[0]*b.a[1] - a.a[1]*b.a[0];
}

template <typename scalar_t>
inline HOST_DEVICE Vector<scalar_t, 3> cross(const Vector<scalar_t, 3> &a, const Vector<scalar_t, 3> &b){
    return {.a={
		a.a[1]*b.a[2] - a.a[2]*b.a[1],
		a.a[2]*b.a[0] - a.a[0]*b.a[2],
		a.a[0]*b.a[1] - a.a[1]*b.a[0]
	}};
}

// MatrixSquare

template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE MatrixSquare<scalar_t, SIZE> operator*(MatrixSquare<scalar_t, SIZE> a, const scalar_t &s){
    for(index_t i=0; i<SIZE; ++i){
        for(index_t j=0; j<SIZE; ++j){
            a.a[i][j] *= s;
        }
    }
    return a;
}

template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE MatrixSquare<scalar_t, SIZE>& operator*=(MatrixSquare<scalar_t, SIZE> &a, const scalar_t &s){
    for(index_t i=0; i<SIZE; ++i){
        for(index_t j=0; j<SIZE; ++j){
            a.a[i][j] *= s;
        }
    }
    return a;
}


template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE MatrixSquare<scalar_t, SIZE> transposed(MatrixSquare<scalar_t, SIZE> &a){
	MatrixSquare<scalar_t, SIZE> b;
	for(index_t i=0; i<SIZE; ++i){
        for(index_t j=0; j<SIZE; ++j){
            b.a[i][j] = a.a[j][i];
        }
    }
	return b;
}

template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE scalar_t mag2(const Vector<scalar_t, SIZE> &a){
    return dot(a,a);
}

template <typename scalar_t>
inline HOST_DEVICE scalar_t _sqrtT(const scalar_t &x);
template <>
inline HOST_DEVICE float _sqrtT<float>(const float &x){
	return sqrtf(x);
}
template <>
inline HOST_DEVICE double _sqrtT<double>(const double &x){
	return sqrt(x);
}

template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE scalar_t mag(const Vector<scalar_t, SIZE> &a){
    return _sqrtT<scalar_t>(mag2(a));
}

template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE Vector<scalar_t, SIZE> matmul(const MatrixSquare<scalar_t, SIZE> &m, const Vector<scalar_t, SIZE> &v){
    Vector<scalar_t, SIZE> result;
    for(index_t row=0; row<SIZE; ++row){
        result.a[row] = dot(m.v[row], v);
        //result.a[row]=0;
        /*for(index_t col=0; col<SIZE; ++col){
            result.a[row] += m.a[row][col] * v.a[col];
        }*/
    }
    return result;
}

template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE Vector<scalar_t, SIZE> matmul(const Vector<scalar_t, SIZE> &v, const MatrixSquare<scalar_t, SIZE> &m){
    Vector<scalar_t, SIZE> result = {.a={0}};
    for(index_t row=0; row<SIZE; ++row){
        for(index_t col=0; col<SIZE; ++col){
            result.a[col] += m.a[row][col] * v.a[row];
        }
    }
    return result;
}

template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE MatrixSquare<scalar_t, SIZE> matmul(const MatrixSquare<scalar_t, SIZE> &mA, const MatrixSquare<scalar_t, SIZE> &mB){
    MatrixSquare<scalar_t, SIZE> result = {.a={0}};
    for(index_t row=0; row<SIZE; ++row){
        result.v[row] = matmul(mA.v[row], mB);
    }
    return result;
}

//template <typename scalar_t, index_t SIZE>
//inline scalar_t det(const MatrixSquare<scalar_t, SIZE> &m);
//https://stackoverflow.com/questions/983999/simple-3x3-matrix-inverse-code-c

template <typename scalar_t>
inline HOST_DEVICE scalar_t det(const MatrixSquare<scalar_t, 1> &m){
    return m.a[0][0];
}
template <typename scalar_t>
inline HOST_DEVICE scalar_t det(const MatrixSquare<scalar_t, 2> &m){
    return m.a[0][0]*m.a[1][1] - m.a[1][0]*m.a[0][1];
}
template <typename scalar_t>
inline HOST_DEVICE scalar_t det(const MatrixSquare<scalar_t, 3> &m){
    return m.a[0][0] * (m.a[1][1]*m.a[2][2] - m.a[2][1]*m.a[1][2])
          -m.a[0][1] * (m.a[1][0]*m.a[2][2] - m.a[2][0]*m.a[1][2])
          +m.a[0][2] * (m.a[1][0]*m.a[2][1] - m.a[2][0]*m.a[1][1]);
}

//template <typename scalar_t, index_t SIZE>
//inline MatrixSquare<scalar_t, SIZE> adjoint(const MatrixSquare<scalar_t, SIZE> &m);

template <typename scalar_t>
inline HOST_DEVICE MatrixSquare<scalar_t, 1> adjoint(const MatrixSquare<scalar_t, 1> &m){
    MatrixSquare<scalar_t, 1> a;
    a.a[0][0]=1; //TODO
    return a;
}
template <typename scalar_t>
inline HOST_DEVICE MatrixSquare<scalar_t, 2> adjoint(const MatrixSquare<scalar_t, 2> &m){
    MatrixSquare<scalar_t, 2> a;
    a.a[0][0]=m.a[1][1];
    a.a[0][1]=-m.a[0][1];
    a.a[1][0]=-m.a[1][0];
    a.a[1][1]=m.a[0][0];
    return a;
}
template <typename scalar_t>
inline HOST_DEVICE MatrixSquare<scalar_t, 3> adjoint(const MatrixSquare<scalar_t, 3> &m){
    MatrixSquare<scalar_t, 3> a;

	// transposed cofactor matrix
    a.a[0][0] = m.a[1][1]*m.a[2][2] - m.a[2][1]*m.a[1][2];
    a.a[0][1] = m.a[0][2]*m.a[2][1] - m.a[0][1]*m.a[2][2];
    a.a[0][2] = m.a[0][1]*m.a[1][2] - m.a[0][2]*m.a[1][1];

    a.a[1][0] = m.a[1][2]*m.a[2][0] - m.a[1][0]*m.a[2][2];
    a.a[1][1] = m.a[0][0]*m.a[2][2] - m.a[0][2]*m.a[2][0];
    a.a[1][2] = m.a[1][0]*m.a[0][2] - m.a[0][0]*m.a[1][2];

    a.a[2][0] = m.a[1][0]*m.a[2][1] - m.a[2][0]*m.a[1][1];
    a.a[2][1] = m.a[2][0]*m.a[0][1] - m.a[0][0]*m.a[2][1];
    a.a[2][2] = m.a[0][0]*m.a[1][1] - m.a[1][0]*m.a[0][1];

	

    return a;
}

template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE MatrixSquare<scalar_t, SIZE> inverse(const MatrixSquare<scalar_t, SIZE> &m){
    MatrixSquare<scalar_t, SIZE> a = adjoint(m);
    a *= 1/det(m);
    return a;
}


template <typename scalar_t>
inline HOST_DEVICE MatrixSquare<scalar_t, 4> inverse(const MatrixSquare<scalar_t, 4> &m){
    scalar_t Coef00 = m.a[2][2] * m.a[3][3] - m.a[3][2] * m.a[2][3];
	scalar_t Coef02 = m.a[1][2] * m.a[3][3] - m.a[3][2] * m.a[1][3];
	scalar_t Coef03 = m.a[1][2] * m.a[2][3] - m.a[2][2] * m.a[1][3];
				   
	scalar_t Coef04 = m.a[2][1] * m.a[3][3] - m.a[3][1] * m.a[2][3];
	scalar_t Coef06 = m.a[1][1] * m.a[3][3] - m.a[3][1] * m.a[1][3];
	scalar_t Coef07 = m.a[1][1] * m.a[2][3] - m.a[2][1] * m.a[1][3];
				   
	scalar_t Coef08 = m.a[2][1] * m.a[3][2] - m.a[3][1] * m.a[2][2];
	scalar_t Coef10 = m.a[1][1] * m.a[3][2] - m.a[3][1] * m.a[1][2];
	scalar_t Coef11 = m.a[1][1] * m.a[2][2] - m.a[2][1] * m.a[1][2];
				   
	scalar_t Coef12 = m.a[2][0] * m.a[3][3] - m.a[3][0] * m.a[2][3];
	scalar_t Coef14 = m.a[1][0] * m.a[3][3] - m.a[3][0] * m.a[1][3];
	scalar_t Coef15 = m.a[1][0] * m.a[2][3] - m.a[2][0] * m.a[1][3];
				   
	scalar_t Coef16 = m.a[2][0] * m.a[3][2] - m.a[3][0] * m.a[2][2];
	scalar_t Coef18 = m.a[1][0] * m.a[3][2] - m.a[3][0] * m.a[1][2];
	scalar_t Coef19 = m.a[1][0] * m.a[2][2] - m.a[2][0] * m.a[1][2];
				   
	scalar_t Coef20 = m.a[2][0] * m.a[3][1] - m.a[3][0] * m.a[2][1];
	scalar_t Coef22 = m.a[1][0] * m.a[3][1] - m.a[3][0] * m.a[1][1];
	scalar_t Coef23 = m.a[1][0] * m.a[2][1] - m.a[2][0] * m.a[1][1];
	
	
	Vector<scalar_t, 4> Fac0 = {.a={Coef00, Coef00, Coef02, Coef03}};
	Vector<scalar_t, 4> Fac1 = {.a={Coef04, Coef04, Coef06, Coef07}};
	Vector<scalar_t, 4> Fac2 = {.a={Coef08, Coef08, Coef10, Coef11}};
	Vector<scalar_t, 4> Fac3 = {.a={Coef12, Coef12, Coef14, Coef15}};
	Vector<scalar_t, 4> Fac4 = {.a={Coef16, Coef16, Coef18, Coef19}};
	Vector<scalar_t, 4> Fac5 = {.a={Coef20, Coef20, Coef22, Coef23}};

	Vector<scalar_t, 4> Vec0 = {.a={m.a[1][0], m.a[0][0], m.a[0][0], m.a[0][0]}};
	Vector<scalar_t, 4> Vec1 = {.a={m.a[1][1], m.a[0][1], m.a[0][1], m.a[0][1]}};
	Vector<scalar_t, 4> Vec2 = {.a={m.a[1][2], m.a[0][2], m.a[0][2], m.a[0][2]}};
	Vector<scalar_t, 4> Vec3 = {.a={m.a[1][3], m.a[0][3], m.a[0][3], m.a[0][3]}};

	Vector<scalar_t, 4> Inv0 = Vec1 * Fac0 - Vec2 * Fac1 + Vec3 * Fac2;
	Vector<scalar_t, 4> Inv1 = Vec0 * Fac0 - Vec2 * Fac3 + Vec3 * Fac4;
	Vector<scalar_t, 4> Inv2 = Vec0 * Fac1 - Vec1 * Fac3 + Vec3 * Fac5;
	Vector<scalar_t, 4> Inv3 = Vec0 * Fac2 - Vec1 * Fac4 + Vec2 * Fac5;

	Vector<scalar_t, 4> SignA = {.a={+1, -1, +1, -1}};
	Vector<scalar_t, 4> SignB = {.a={-1, +1, -1, +1}};
	MatrixSquare<scalar_t, 4> Inverse = {.v={Inv0 * SignA, Inv1 * SignB, Inv2 * SignA, Inv3 * SignB}};

	Vector<scalar_t, 4> Row0 = {.a={Inverse.a[0][0], Inverse.a[1][0], Inverse.a[2][0], Inverse.a[3][0]}};

	scalar_t Dot1 = dot(m.v[0], Row0); //(m[0] * Row0);
	//T Dot1 = (Dot0.x + Dot0.y) + (Dot0.z + Dot0.w);

	scalar_t OneOverDeterminant = 1.f / Dot1;

	return Inverse * OneOverDeterminant;
}

template <typename scalar_t, index_t SIZE>
inline HOST_DEVICE MatrixSquare<scalar_t, SIZE> inverse(const MatrixSquare<scalar_t, SIZE> &m, const scalar_t &det){
    MatrixSquare<scalar_t, SIZE> a = adjoint(m);
    a *= 1/det;
    return a;
}

template <typename scalar_t>
HOST_DEVICE inline
scalar_t lerp(const scalar_t a, const scalar_t b, const scalar_t t){
	return (1.0-t)*a + t*b;
}
template <typename scalar_t, int DIMS>
HOST_DEVICE inline
Vector<scalar_t,DIMS> lerp(const Vector<scalar_t,DIMS> &a, const Vector<scalar_t,DIMS> &b, const scalar_t t){
	return a*(1.0-t) + b*t;
}

template <typename scalar_t, int DIMS>
HOST_DEVICE inline
Vector<scalar_t,DIMS> slerp1(const Vector<scalar_t,DIMS> &v1, const Vector<scalar_t,DIMS> &v2, const scalar_t t, const scalar_t eps){
	// normalize vectors
	const scalar_t sqlen1 = dot(v1,v1);
	const scalar_t sqlen2 = dot(v2,v2);
	if(sqlen1<eps || sqlen2<eps){
		// vector to short to normalize, default to lerp
		return lerp(v1,v2,t);
	}
	const Vector<scalar_t,DIMS> n1 = v1*rsqrt(sqlen1);
	const Vector<scalar_t,DIMS> n2 = v2*rsqrt(sqlen1);
	
	// get angle between vectors
	const scalar_t rad = acos(dot(n1, n2));
	if(rad<eps){
		return lerp(v1,v2,t);
	}
	
	// slerp direction and magnitude
	const scalar_t w1 = sin((1.0-t)*rad) / sin(rad);
	const scalar_t w2 = sin(t*rad) / sin(rad);
	const Vector<scalar_t,DIMS> v = v1*w1 + v2*w2;
	
	return v;
}

template <typename scalar_t, int DIMS>
HOST_DEVICE inline
Vector<scalar_t,DIMS> slerp2(const Vector<scalar_t,DIMS> &v1, const Vector<scalar_t,DIMS> &v2, const scalar_t t, const scalar_t eps){
	// normalize vectors
	const scalar_t sqlen1 = dot(v1,v1);
	const scalar_t sqlen2 = dot(v2,v2);
	if(sqlen1<eps || sqlen2<eps){
		// vector to short to normalize, default to lerp
		return lerp(v1,v2,t);
	}
	const Vector<scalar_t,DIMS> n1 = v1*rsqrt(sqlen1);
	const Vector<scalar_t,DIMS> n2 = v2*rsqrt(sqlen1);
	
	// get angle between vectors
	const scalar_t rad = acos(dot(n1, n2));
	if(rad<eps){
		return lerp(v1,v2,t);
	}
	
	// slerp direction
	const scalar_t w1 = sin((1.0-t)*rad) / sin(rad);
	const scalar_t w2 = sin(t*rad) / sin(rad);
	const Vector<scalar_t,DIMS> n = n1*w1 + n2*w2;
	
	// lerp magnitude
	const scalar_t m1 = sqrt(sqlen1);
	const scalar_t m2 = sqrt(sqlen2);
	const scalar_t m = lerp(m1,m2,t);
	
	const Vector<scalar_t,DIMS> v = n*m;
	
	return v;
}

#ifdef HOST_DEVICE
#undef HOST_DEVICE
#endif

#endif //_INCLUDE_TRANSFORMATIONS