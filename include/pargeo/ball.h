#pragma once

#include <iostream>
#include <algorithm>
#include <math.h>
#include <iomanip>
#include "parlay/parallel.h"
#include "parlay/primitives.h"

namespace pargeo {

  struct _empty {
    int arr[0]; // todo this produces a struct of size 0 but seems dangerous, need to check
    _empty(){}
    
    template<class T>
    _empty(T x){}

    friend bool operator<(_empty a, _empty b) {return true;}
    friend bool operator==(_empty a, _empty b) {return true;}
  };

  template <int _dim, class pointT>
  class _ball {
  public:
  	pointT center;
  	double radius;
  	int count;
  	_ball(pointT& _center, double _radius): center(_center), radius(_radius), count(0){}

  	bool contains_point(pointT& p){
  		return center.distSqr(p) <= radius*radius;
  	}

  	inline pointT pointClosestToCenter(pointT& center, pointT& pMin, pointT& pMax) {
		pointT p;
		for (int d = 0; d < dim; ++ d) {
			p[d] = std::max(pMin[d], std::min(center[d], pMax[d]));
		}
		return p;
	}

  	bool intersect_rect(pointT& minR, pointT& maxR){
  		return center.distSqr(pointClosestToCenter(center, minR, maxR)) < radius*radius;
  	}
  };  
}

