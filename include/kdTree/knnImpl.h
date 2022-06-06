// This code is part of the project "ParGeo: A Library for Parallel Computational Geometry"
// Copyright (c) 2021-2022 Yiqiu Wang, Shangdi Yu, Laxman Dhulipala, Yan Gu, Julian Shun
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <limits>
#include <algorithm>
#include "parlay/parallel.h"
#include "parlay/sequence.h"
#include "kdTree.h"
#include "pargeo/point.h"

namespace pargeo::kdTree
{

	
	template <int dim, typename nodeT, typename objT>
	void knnRange(nodeT *tree, objT &q, double &radius, objT *&out);

	template <int dim, typename nodeT, typename objT>
	void knnHelper(nodeT *tree, objT &q, objT *&out);

	template <int dim, typename nodeT, typename objT>
	void knnRangeHelper(nodeT *tree, objT &q, objT qMin, objT qMax, double &radius, objT *&out);


	template <int dim, typename nodeT, typename objT>
	void knnHelper(nodeT *tree, objT &q, double &radius, objT *&out){
		// find the leaf first
		int relation = tree->boxCompare(tree->getMin(), tree->getMax(),
										point<dim>(q.coords()),
										point<dim>(q.coords()));
//		std::cout<<q[0]<<" "<<q[1]<<" : "<<tree->getNBMin()[0]<<" "<<tree->getNBMin()[1]<<"; "<<tree->getNBMax()[0]<<" "<<tree->getNBMax()[1]<<std::endl;
		if (relation == tree->boxExclude)
		{
			return;
		}
		else
		{
			if (tree->isLeaf())
			{
				// basecase
				for (size_t i = 0; i < tree->size(); ++i)
				{
					objT *p = tree->getItem(i);
					double dist = q.dist(*p);
					if(dist < radius){
						radius = dist;
						out = p;
					}
				}
			}
			else
			{
				knnHelper<dim, nodeT, objT>(tree->L(), q, radius, out);
				knnHelper<dim, nodeT, objT>(tree->R(), q, radius, out);
			}
		}

		if(tree->siblin() != NULL)
		{
			knnRange<dim, nodeT, objT>(tree->siblin(), q, radius, out);
		}
	}
	
	

	template <int dim, typename nodeT, typename objT>
	void knnRangeHelper(nodeT *tree, objT &q, objT qMin, objT qMax, 
						double &radius, objT *&out)
	{
		int relation = tree->boxCompare(qMin, qMax, tree->getMin(), tree->getMax());
		if (relation == tree->boxExclude)
		{
			return;
		}
		else if (relation == tree->boxInclude) // first includes second, target region includes tree region
		{ // use threshold to decide going down vs bruteforce
			if (tree->isLeaf())
			{
				for(size_t i=0; i<tree->size(); ++i)
				{
					objT *p = tree->getItem(i);
					double dist = q.dist(*p);
					if(dist < radius){
						radius = dist;
						out = p;
					}
				}
			}
			else
			{
				knnRangeHelper<dim, nodeT, objT>(tree->L(), q, qMin, qMax, radius, out);
				knnRange<dim, nodeT, objT>(tree->R(), q, radius, out);
			}
		}
		else
		{ // intersect
			if (tree->isLeaf())
			{
				for(size_t i=0; i<tree->size(); ++i)
				{
					objT *p = tree->getItem(i);
					double dist = q.dist(*p);
					if(dist < radius){
						radius = dist;
						out = p;
					}
				}
			}
			else
			{
				knnRangeHelper<dim, nodeT, objT>(tree->L(), q, qMin, qMax, radius, out);
				knnRange<dim, nodeT, objT>(tree->R(), q, radius, out);
			}
		}
	}

	template <int dim, typename nodeT, typename objT>
	void knnRange(nodeT *tree, objT &q, double &radius, objT *&out)
	{
		objT qMin, qMax;
		for (size_t i = 0; i < dim; i++)
		{
			auto tmp = q[i] - radius;
			qMin[i] = tmp;
			qMax[i] = tmp + radius * 2;
		}
		knnRangeHelper<dim, nodeT, objT>(tree, q, qMin, qMax, radius, out);
	}

	
	template <int dim, class objT>
	objT* NearestNeighbor(objT &q, node<dim, objT> *tree){
		using nodeT = node<dim, objT>;
		double radius = std::numeric_limits<double>::max()/2;
		objT* out = NULL;
		knnHelper<dim, nodeT, objT>(tree, q, radius, out);
		return out;
	}



} // End namespace pargeo
