#pragma once

#include <iostream>
#include <algorithm>
#include <math.h>
#include <iomanip>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "pargeo/atomics.h"

namespace pargeo {

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

  	inline void count_increase(int number){
  		pargeo::write_add(&count, number);
  	}

  	inline pointT pointClosestToCenter(pointT& center, pointT& pMin, pointT& pMax) {
		pointT p;
		for (int d = 0; d < _dim; ++ d) {
			p[d] = std::max(pMin[d], std::min(center[d], pMax[d]));
		}
		return p;
	}

	inline pointT pointFarthestFromCenter(pointT center, pointT& pMin, pointT& pMax){
		pointT p;
		for (int d = 0; d < _dim; ++ d){
			p[d] = (center[d]*2 < pMin[d]+pMax[d]) ? pMax[d] : pMin[d]; 
		}
		return p;
	}

  	bool intersect_rect(pointT& minR, pointT& maxR){
  		return center.distSqr(pointClosestToCenter(center, minR, maxR)) < radius*radius;
  	}

  	bool contains_rect(pointT& minR, pointT& maxR){
  		return center.distSqr(pointFarthestFromCenter(center, minR, maxR)) < radius*radius;
  	}
  };  
}

