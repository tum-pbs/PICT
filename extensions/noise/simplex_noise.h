#pragma once

#ifndef _INCLUDE_SIMPLEX_NOISE
#define _INCLUDE_SIMPLEX_NOISE

#include "../transformations.h"
/*
https://forum.unity.com/threads/2d-3d-4d-optimised-perlin-noise-cg-hlsl-library-cginc.218372/
Description:
	Array- and textureless CgFx/HLSL 2D, 3D and 4D simplex noise functions.
	a.k.a. simplified and optimized Perlin noise.
	
	The functions have very good performance
	and no dependencies on external data.
	
	2D - Very fast, very compact code.
	3D - Fast, compact code.
	4D - Reasonably fast, reasonably compact code.

------------------------------------------------------------------

Ported by:
	I've aditionally ported the port to CUDA.
	
	Lex-DRL
	I've ported the code from GLSL to CgFx/HLSL for Unity,
	added a couple more optimisations (to speed it up even further)
	and slightly reformatted the code to make it more readable.

Original GLSL functions:
	https://github.com/ashima/webgl-noise
	Credits from original glsl file are at the end of this cginc.

------------------------------------------------------------------

Usage:
	
	float ns = snoise(v);
	// v is any of: Vector<float,2>, Vector<float,3>, Vector<float,4>
	
	Return type is float.
	To generate 2 or more components of noise (colorful noise),
	call these functions several times with different
	constant offsets for the arguments.
	E.g.:
	
	Vector<float,3> colorNs = Vector<float,3>(
		snoise(v),
		snoise(v + 17.0),
		snoise(v - 43.0),
	);


Remark about those offsets from the original author:
	
	People have different opinions on whether these offsets should be integers
	for the classic noise functions to match the spacing of the zeroes,
	so we have left that for you to decide for yourself.
	For most applications, the exact offsets don't really matter as long
	as they are not too small or too close to the noise lattice period
	(289 in this implementation).

*/

namespace SimplexNoise{

namespace vmath{

// template <typename scalar_t, size_t SIZE>
// inline __device__
// Vector<scalar_t, SIZE> floor(Vector<scalar_t, SIZE> v);

template <size_t SIZE>
inline __device__
Vector<float, SIZE> floor(Vector<float, SIZE> v){
    for(size_t i=0; i<SIZE; ++i){
        v.a[i] = floorf(v.a[i]);
    }
    return v;
}
template <size_t SIZE>
inline __device__
Vector<double, SIZE> floor(Vector<double, SIZE> v){
    for(size_t i=0; i<SIZE; ++i){
        v.a[i] = floor(v.a[i]);
    }
    return v;
}


// template <typename scalar_t, size_t SIZE>
// inline __device__
// Vector<scalar_t, SIZE> frac(Vector<scalar_t, SIZE> v);

template <size_t SIZE>
inline __device__
Vector<float, SIZE> frac(Vector<float, SIZE> v){
    for(size_t i=0; i<SIZE; ++i){
        v.a[i] = v.a[i] - floorf(v.a[i]);
    }
    return v;
}
template <size_t SIZE>
inline __device__
Vector<double, SIZE> frac(Vector<double, SIZE> v){
    for(size_t i=0; i<SIZE; ++i){
        v.a[i] = v.a[i] - floor(v.a[i]);
    }
    return v;
}

// template <typename scalar_t, size_t SIZE>
// inline __device__
// Vector<scalar_t, SIZE> fabs(Vector<scalar_t, SIZE> v);

template <size_t SIZE>
inline __device__
Vector<float, SIZE> fabs(Vector<float, SIZE> v){
    for(size_t i=0; i<SIZE; ++i){
        v.a[i] = fabsf(v.a[i]);
    }
    return v;
}
template <size_t SIZE>
inline __device__
Vector<double, SIZE> fabs(Vector<double, SIZE> v){
    for(size_t i=0; i<SIZE; ++i){
        v.a[i] = fabs(v.a[i]);
    }
    return v;
}

template <size_t SIZE>
inline __device__
Vector<float, SIZE> fmin(Vector<float, SIZE> v, const Vector<float, SIZE> b){
    for(size_t i=0; i<SIZE; ++i){
        v.a[i] = fminf(v.a[i], b.a[i]);
    }
    return v;
}
template <size_t SIZE>
inline __device__
Vector<double, SIZE> fmin(Vector<double, SIZE> v, const Vector<float, SIZE> b){
    for(size_t i=0; i<SIZE; ++i){
        v.a[i] = fmin(v.a[i], b.a[i]);
    }
    return v;
}

/* template <typename scalar_t, size_t SIZE>
inline __device__
Vector<scalar_t, SIZE> fmax(Vector<scalar_t, SIZE> v, Vector<scalar_t, SIZE> &b); */

template <size_t SIZE>
inline __device__
Vector<float, SIZE> fmax(Vector<float, SIZE> v, const Vector<float, SIZE> b){
    for(size_t i=0; i<SIZE; ++i){
        v.a[i] = fmaxf(v.a[i], b.a[i]);
    }
    return v;
}
template <size_t SIZE>
inline __device__
Vector<double, SIZE> fmax(Vector<double, SIZE> v, const Vector<float, SIZE> b){
    for(size_t i=0; i<SIZE; ++i){
        v.a[i] = fmax(v.a[i], b.a[i]);
    }
    return v;
}

} //vmath

// 1 / 289
#define NOISE_SIMPLEX_1_DIV_289 0.00346020761245674740484429065744f

inline __device__ float mod289(const float x) {
	return x - floorf(x * NOISE_SIMPLEX_1_DIV_289) * 289.0f;
}

inline __device__ Vector<float,2> mod289(const Vector<float,2> x) {
	return x - vmath::floor<2>(x * NOISE_SIMPLEX_1_DIV_289) * 289.0f;
}

inline __device__ Vector<float,3> mod289(const Vector<float,3> x) {
	return x - vmath::floor<3>(x * NOISE_SIMPLEX_1_DIV_289) * 289.0f;
}

inline __device__ Vector<float,4> mod289(const Vector<float,4> x) {
	return x - vmath::floor<4>(x * NOISE_SIMPLEX_1_DIV_289) * 289.0f;
}


// ( x*34.0 + 1.0 )*x = 
// x*x*34.0 + x
inline __device__ float permute(const float x) {
	return mod289(
		x*x*34.0f + x
	);
}

inline __device__ Vector<float,3> permute(const Vector<float,3> x) {
	return mod289(
		x*x*34.0f + x
	);
}

inline __device__ Vector<float,4> permute(const Vector<float,4> x) {
	return mod289(
		x*x*34.0f + x
	);
}



constexpr __device__ float taylorInvSqrt(const float r) {
	return 1.79284291400159f - 0.85373472095314f * r;
}

inline __device__ Vector<float,4> taylorInvSqrt(const Vector<float,4> r) {
	return 1.79284291400159f - 0.85373472095314f * r;
}



inline __device__ Vector<float,4> grad4(const float j, const Vector<float,4> ip) {
//	const Vector<float,4> ones = make_Vector<float,4>(1.0, 1.0, 1.0, -1.0);
//	Vector<float,4> p, s;
//	p.xyz = floor( fracf(j * ip.xyz) * 7.0) * ip.z - 1.0;
//	p.w = 1.5 - dot( fabs(p.xyz), ones.xyz );
	// Vector<float,4> p = make_Vector<float,4>(
		// floorf( fracf(j * SW3_F(ip,x,y,z)) * 7.0f) * ip.z - 1.0f, //xyz
		// 0.f //w
	// );
	Vector<float,4> p = vmath::floor<4>(vmath::frac<4>(j * ip) * 7.0f) * ip.z - 1.0f;
	p.w = 1.5f - fabs(p.x)+fabs(p.y)+fabs(p.z);
	
	// GLSL: lessThan(x, y) = x < y
	// HLSL: 1 - vmath::step(y, x) = x < y
	const Vector<float,4> s = p < 0.0f; //1.f - vmath::step(make_Vector<float,4>(0.0f), p);
	p.x += (s.x * 2.f - 1.f) * s.w; 
	p.y += (s.y * 2.f - 1.f) * s.w; 
	p.z += (s.z * 2.f - 1.f) * s.w; 
	
	return p;
}



// ----------------------------------- 2D -------------------------------------

inline __device__ float snoise(const Vector<float,2> v)
{
	const Vector<float,4> C{.a={
		0.211324865405187f, // (3.0-sqrt(3.0))/6.0
		0.366025403784439f, // 0.5*(sqrt(3.0)-1.0)
	   -0.577350269189626f, // -1.0 + 2.0 * C.x
		0.024390243902439f  // 1.0 / 41.0
	}};
	
// First corner
	Vector<float,2> i = vmath::floor<2>( v + (v.x*C.y + v.y*C.y));//dot(v, SW2_F(C,y,y)) );
	Vector<float,2> x0 = v - i + (i.x*C.x + i.y*C.x);//dot(i, SW2_F(C,x,x));
	
// Other corners
	// Vector<float,2> i1 = (x0.x > x0.y) ? Vector<float,2>(1.0, 0.0) : Vector<float,2>(0.0, 1.0);
	// Lex-DRL: afaik, vmath::step() in GPU is faster than if(), so:
	// vmath::step(x, y) = x <= y
	float xLessEqual = x0.x <= x0.y; // x <= y ?
	Vector<float,2> i1{.a={1.0f - xLessEqual, xLessEqual}};
	/* int2 i1 =
		make_int2(1, 0) * (1 - xLessEqual) // x > y
		+ make_int2(0, 1) * xLessEqual // x <= y
	; */
	//Vector<float,4> x12 = SW4_F(x0,x,y,x,y) + SW4_F(C,x,x,z,z);
	Vector<float,2> x1{.a={x0.x + C.x, x0.y + C.x}};
	Vector<float,2> x2{.a={x0.x + C.z, x0.y + C.z}};
	x1 -= i1;
	
// Permutations
	i = mod289(i); // Avoid truncation effects in permutation
	Vector<float,3> p = permute(
		permute(
				i.y + Vector<float,3>{.a={0.0f, i1.y, 1.0f}}
		) + i.x + Vector<float,3>{.a={0.0f, i1.x, 1.0f}}
	);
	
	Vector<float,3> m = vmath::fmax<3>(
		0.5f - Vector<float,3>{.a={
			dot(x0, x0),
			dot(x1, x1),//dot(x12.xy, x12.xy),
			dot(x2, x2)//dot(x12.zw, x12.zw)
		}},
		Vector<float,3>{.a={0.0f}}
	);
	m = m*m ;
	m = m*m ;
	
// Gradients: 41 points uniformly over a line, mapped onto a diamond.
// The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)
	
	Vector<float,3> x = 2.0f * vmath::frac<3>(p * C.w) - 1.0f;
	Vector<float,3> h = vmath::fabs<3>(x) - 0.5f;
	Vector<float,3> ox = vmath::floor<3>(x + 0.5f);
	Vector<float,3> a0 = x - ox;

// Normalise gradients implicitly by scaling m
// Approximation of: m *= inversesqrt( a0*a0 + h*h );
	m *= 1.79284291400159f - 0.85373472095314f * ( a0*a0 + h*h );

// Compute final noise value at P
	Vector<float,3> g{.a={
		a0.x * x0.x + h.x * x0.y,
		a0.y * x1.x + h.y * x1.y,
		a0.z * x2.x + h.z * x2.y
	}};
	return 130.0f * dot(m, g);
}

// ----------------------------------- 3D -------------------------------------

inline __device__ float snoise(const Vector<float,3> v)
{
	const Vector<float,2> C{
		0.166666666666666667f, // 1/6
		0.333333333333333333f  // 1/3
	};
	const Vector<float,4> D{0.0f, 0.5f, 1.0f, 2.0f};
	
// First corner
	Vector<float,3> i = vmath::floor<3>( v + (v.x*C.y + v.y*C.y + v.z*C.y));//dot(v, C.yyy) );
	Vector<float,3> x0 = v - i + (i.x*C.x + i.y*C.x + i.z*C.x);//dot(i, C.xxx);
	
// Other corners
	Vector<float,3> g = (Vector<float,3>{x0.y,x0.z,x0.x} <= x0);//vmath::step(x0.yzx, x0.xyz);
	Vector<float,3> l = 1.0f - g;
	Vector<float,3> i1 = vmath::fmin<3>(g, Vector<float,3>{l.z,l.x,l.y});//min(g.xyz, l.zxy);
	Vector<float,3> i2 = vmath::fmax<3>(g, Vector<float,3>{l.z,l.x,l.y});//max(g.xyz, l.zxy);
	
	Vector<float,3> x1 = x0 - i1 + C.x;
	Vector<float,3> x2 = x0 - i2 + C.y; // 2.0*C.x = 1/3 = C.y
	Vector<float,3> x3 = x0 - D.y;      // -1.0+3.0*C.x = -0.5 = -D.y
	
// Permutations
	i = mod289(i);
	Vector<float,4> p = permute(
		permute(
			permute(
					i.z + Vector<float,4>{0.0f, i1.z, i2.z, 1.0f}
			) + i.y + Vector<float,4>{0.0f, i1.y, i2.y, 1.0f}
		) 	+ i.x + Vector<float,4>{0.0f, i1.x, i2.x, 1.0f}
	);
	
// Gradients: 7x7 points over a square, mapped onto an octahedron.
// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
	const float n_ = 0.142857142857f; // 1/7
	Vector<float,3> ns{ //n_ * D.wyz - D.xzx;
		n_* D.w - D.x,
		n_* D.y - D.z,
		n_* D.z - D.x
	};
	
	Vector<float,4> j = p - 49.0f * vmath::floor<4>(p * ns.z * ns.z); // mod(p,7*7)
	
	Vector<float,4> x_ = vmath::floor<4>(j * ns.z);
	Vector<float,4> y_ = vmath::floor<4>(j - 7.0f * x_ ); // mod(j,N)
	
	Vector<float,4> x = x_ *ns.x + ns.y;
	Vector<float,4> y = y_ *ns.x + ns.y;
	Vector<float,4> h = 1.0f - vmath::fabs<4>(x) - vmath::fabs<4>(y);
	
	Vector<float,4> b0{x.x, x.y, y.x, y.y};
	Vector<float,4> b1{x.z, x.w, y.z, y.w};
	
	//Vector<float,4> s0 = Vector<float,4>(lessThan(b0,0.0))*2.0 - 1.0;
	//Vector<float,4> s1 = Vector<float,4>(lessThan(b1,0.0))*2.0 - 1.0;
	Vector<float,4> s0 = vmath::floor<4>(b0)*2.0f + 1.0f;
	Vector<float,4> s1 = vmath::floor<4>(b1)*2.0f + 1.0f;
	Vector<float,4> sh = 0.0f - (h <= 0.0f);
	
	Vector<float,4> a0{ //b0.xzyw + s0.xzyw*sh.xxyy ;
		b0.x + s0.x*sh.x,
		b0.z + s0.z*sh.x,
		b0.y + s0.y*sh.y,
		b0.w + s0.w*sh.y
	};
	Vector<float,4> a1{ //b1.xzyw + s1.xzyw*sh.zzww ;
		b1.x + s1.x*sh.z,
		b1.z + s1.z*sh.z,
		b1.y + s1.y*sh.w,
		b1.w + s1.w*sh.w
	};
	
	Vector<float,3> p0{a0.x, a0.y, h.x};
	Vector<float,3> p1{a0.z, a0.w, h.y};
	Vector<float,3> p2{a1.x, a1.y, h.z};
	Vector<float,3> p3{a1.z, a1.w, h.w};
	
//Normalise gradients
	Vector<float,4> norm = taylorInvSqrt(Vector<float,4>{
		dot(p0, p0),
		dot(p1, p1),
		dot(p2, p2),
		dot(p3, p3)
	});
	p0 *= norm.x;
	p1 *= norm.y;
	p2 *= norm.z;
	p3 *= norm.w;
	
// Mix final noise value
	Vector<float,4> m = vmath::fmax<4>(
		0.6f - Vector<float,4>{
			dot(x0, x0),
			dot(x1, x1),
			dot(x2, x2),
			dot(x3, x3)
		},
		Vector<float,4>{0.0f}
	);
	m = m * m;
	return 42.0f * dot(
		m*m,
		Vector<float,4>{
			dot(p0, x0),
			dot(p1, x1),
			dot(p2, x2),
			dot(p3, x3)
		}
	);
}

// ----------------------------------- 4D -------------------------------------
/*
inline __device__ float snoise(const Vector<float,4> v)
{
	const Vector<float,4> C = make_Vector<float,4>(
		0.138196601125011f, // (5 - sqrt(5))/20 G4
		0.276393202250021f, // 2 * G4
		0.414589803375032f, // 3 * G4
	 -0.447213595499958f  // -1 + 4 * G4
	);

// First corner
	Vector<float,4> i = floorf(
		v +
		dot(
			v,
			make_Vector<float,4>(0.309016994374947451f) // (sqrt(5) - 1) / 4
		)
	);
	Vector<float,4> x0 = v - i + vmath::sum(i*C.x);//dot(i, C.xxxx);

// Other corners

// Rank sorting originally contributed by Bill Licea-Kane, AMD (formerly ATI)
	Vector<float,3> isX = vmath::step( SW3_F(x0,y,z,w), SW3_F(x0,x,x,x) );
	Vector<float,3> isYZ = vmath::step( SW3_F(x0,z,w,w), SW3_F(x0,y,y,z) );
	// Vector<float,4> i0;
	// i0.x = isX.x + isX.y + isX.z;
	// i0.yzw = 1.0 - isX;
	// i0.y += isYZ.x + isYZ.y;
	// i0.zw += 1.0 - isYZ.xy;
	// i0.z += isYZ.z;
	// i0.w += 1.0 - isYZ.z;

	Vector<float,4> i0 = make_Vector<float,4>(
		isX.x + isX.y + isX.z,
		1.0f - (isX.x) + isYZ.x + isYZ.y,
		1.0f - (isX.y + isYZ.x) + isYZ.z,
		1.0f - (isX.y + isYZ.x + isYZ.z)
	);

	// i0 now contains the unique values 0,1,2,3 in each channel
	Vector<float,4> i3 = clamp(i0, 0.0f, 1.0f);//saturate(i0);
	Vector<float,4> i2 = clamp(i0-1.0f, 0.0f, 1.0f);//saturate(i0-1.0);
	Vector<float,4> i1 = clamp(i0-2.0f, 0.0f, 1.0f);//saturate(i0-2.0);

	//	x0 = x0 - 0.0 + 0.0 * C.xxxx
	//	x1 = x0 - i1  + 1.0 * C.xxxx
	//	x2 = x0 - i2  + 2.0 * C.xxxx
	//	x3 = x0 - i3  + 3.0 * C.xxxx
	//	x4 = x0 - 1.0 + 4.0 * C.xxxx
	Vector<float,4> x1 = x0 - i1 + C.x;
	Vector<float,4> x2 = x0 - i2 + C.y;
	Vector<float,4> x3 = x0 - i3 + C.z;
	Vector<float,4> x4 = x0 + C.w;

// Permutations
	i = mod289(i); 
	float j0 = permute(
		permute(
			permute(
				permute(i.w) + i.z
			) + i.y
		) + i.x
	);
	Vector<float,4> j1 = permute(
		permute(
			permute(
				permute (
					i.w + make_Vector<float,4>(i1.w, i2.w, i3.w, 1.0f )
				) + i.z + make_Vector<float,4>(i1.z, i2.z, i3.z, 1.0f )
			) + i.y + make_Vector<float,4>(i1.y, i2.y, i3.y, 1.0f )
		) + i.x + make_Vector<float,4>(i1.x, i2.x, i3.x, 1.0f )
	);

// Gradients: 7x7x6 points over a cube, mapped onto a 4-cross polytope
// 7*7*6 = 294, which is close to the ring size 17*17 = 289.
	const Vector<float,4> ip = make_Vector<float,4>(
		0.003401360544217687075f, // 1/294
		0.020408163265306122449f, // 1/49
		0.142857142857142857143f, // 1/7
		0.0f
	);

	Vector<float,4> p0 = grad4(j0, ip);
	Vector<float,4> p1 = grad4(j1.x, ip);
	Vector<float,4> p2 = grad4(j1.y, ip);
	Vector<float,4> p3 = grad4(j1.z, ip);
	Vector<float,4> p4 = grad4(j1.w, ip);

// Normalise gradients
	Vector<float,4> norm = taylorInvSqrt(make_Vector<float,4>(
		dot(p0, p0),
		dot(p1, p1),
		dot(p2, p2),
		dot(p3, p3)
	));
	p0 *= norm.x;
	p1 *= norm.y;
	p2 *= norm.z;
	p3 *= norm.w;
	p4 *= taylorInvSqrt( dot(p4, p4) );

// Mix contributions from the five corners
	Vector<float,3> m0 = fmaxf(
		0.6f - make_Vector<float,3>(
			dot(x0, x0),
			dot(x1, x1),
			dot(x2, x2)
		),
		make_Vector<float,3>(0.0f)
	);
	Vector<float,2> m1 = fmaxf(
		0.6f - make_Vector<float,2>(
			dot(x3, x3),
			dot(x4, x4)
		),
		make_Vector<float,2>(0.0f)
	);
	m0 = m0 * m0;
	m1 = m1 * m1;
	
	return 49.0f * (
		dot(
			m0*m0,
			make_Vector<float,3>(
				dot(p0, x0),
				dot(p1, x1),
				dot(p2, x2)
			)
		) + dot(
			m1*m1,
			make_Vector<float,2>(
				dot(p3, x3),
				dot(p4, x4)
			)
		)
	);
}
*/

} //SimplexNoise


//                 Credits from source glsl file:
//
// Description : Array and textureless GLSL 2D/3D/4D simplex 
//               noise functions.
//      Author : Ian McEwan, Ashima Arts.
//  Maintainer : ijm
//     Lastmod : 20110822 (ijm)
//     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
//               Distributed under the MIT License. See LICENSE file.
//               https://github.com/ashima/webgl-noise
//
//
//           The text from LICENSE file:
//
//
// Copyright (C) 2011 by Ashima Arts (Simplex noise)
// Copyright (C) 2011 by Stefan Gustavson (Classic noise)
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#endif //_INCLUDE_SIMPLEX_NOISE