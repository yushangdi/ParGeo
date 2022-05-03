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
// #include <omp.h>

namespace pargeo::kdTreeNUMA
{

  namespace knnBuf
  {

    typedef int intT;
    typedef double floatT;

    template <typename T>
    struct elem
    {
      floatT cost; // Non-negative
      T entry;
      elem(floatT t_cost, T t_entry) : cost(t_cost), entry(t_entry) {}
      elem() : cost(std::numeric_limits<floatT>::max()) {}
      bool operator<(const elem &b) const
      {
        if (cost < b.cost)
          return true;
        return false;
      }
    };

    template <typename T>
    struct buffer
    {
      typedef parlay::slice<elem<T> *, elem<T> *> sliceT;
      intT k;
      intT ptr;
      sliceT buf;
      double max_cost = 0;

      buffer(intT t_k, sliceT t_buf) : k(t_k), ptr(0), buf(t_buf) {}

      inline void reset() { ptr = 0; }

      bool hasK() { return ptr >= k; }

      elem<T> keepK()
      {
        if (ptr < k)
          throw std::runtime_error("Error, kbuffer not enough k.");
        ptr = k;
        std::nth_element(buf.begin(), buf.begin() + k - 1, buf.end());
        max_cost = 0;
        for(auto b = buf.begin(); b < buf.begin() + k ; ++b){
          max_cost = std::max(max_cost, b->cost);
        }
        return buf[k - 1];
      }

      void sort()
      { // todo check
        if (ptr < k)
          throw std::runtime_error("Error, sorting kbuffer without enough k.");
        parlay::sort(buf.cut(0, k));
      }

      void insert(elem<T> t_elem)
      {
        buf[ptr++] = t_elem;
        max_cost = std::max(max_cost, t_elem.cost);
        if (ptr >= buf.size())
          keepK();
      }

      elem<T> operator[](intT i)
      {
        if (i < ptr)
          return buf[i];
        else
          return elem<T>();
      }

      inline size_t size() {return ptr;}

      inline double back() {
        if(ptr<k){return std::numeric_limits<double>::max();}
        return max_cost;
      }

    };
  }

  template <int dim, typename nodeT, typename objT>
  void knnRangeHelper2(nodeT *tree, objT &q,
                      double radius, knnBuf::buffer<objT *> &out)
  {
    int relation = tree->boxBallCompare(q, radius, tree->getMin(), tree->getMax());

    if (relation == tree->boxExclude)
    {
      return;
    }
    else
    { // intersect
      if (tree->isLeaf())
      {
        for (size_t i = 0; i < tree->size(); ++i)
        {
          objT *p = tree->getItem(i);
          double dist = q.dist(*p);
          if (dist <= radius)
          {
            out.insert(knnBuf::elem(dist, p));
            radius = out.back();
          }
        }
      }
      else
      {
        knnRangeHelper2<dim, nodeT, objT>(tree->L(), q, radius, out);
        radius =  out.back();
        knnRangeHelper2<dim, nodeT, objT>(tree->R(), q, radius, out);
      }
    }
  }

  template <int dim, typename nodeT, typename objT>
  void knnHelper2(nodeT *tree, objT &q, knnBuf::buffer<objT *> &out)
  {
    // find the leaf first
    int relation = tree->boxCompare(tree->getMin(), tree->getMax(),
                                    point<dim>(q.coords()),
                                    point<dim>(q.coords()));
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
          out.insert(knnBuf::elem(q.dist(*p), p));
        }
      }
      else
      {
        knnHelper2<dim, nodeT, objT>(tree->L(), q, out);
        knnHelper2<dim, nodeT, objT>(tree->R(), q, out);
      }
    }

    if (!out.hasK())
    {
      if (tree->siblin() == NULL)
      {
        throw std::runtime_error("Error, knnHelper reached root node without enough neighbors.");
      }
      for (size_t i = 0; i < tree->siblin()->size(); ++i)
      {
        objT *p = tree->siblin()->getItem(i);
        out.insert(knnBuf::elem(q.dist(*p), p));
      }
    }
    else
    { // Buffer filled to a least k
      if (tree->siblin() != NULL)
      {
        knnRangeHelper2<dim, nodeT, objT>(tree->siblin(), q, out.back(), out);
      }
    }
  }

  template <int dim, class objT>
  parlay::sequence<size_t> batchKnn2(parlay::slice<objT *, objT *> queries,
                                    size_t k,
                                    node<dim, objT> *tree,
                                    bool sorted=false)
  {
    using nodeT = node<dim, objT>;
    bool freeTree = false;
    if (!tree)
    {
      freeTree = true;
      tree = build<dim, objT>(queries, true);
    }
    auto out = parlay::sequence<knnBuf::elem<objT *>>(2 * k * queries.size());
    auto idx = parlay::sequence<size_t>(k * queries.size());
    parlay::parallel_for(0, queries.size(), [&](size_t i)
                         {
                           knnBuf::buffer buf = knnBuf::buffer<objT *>(k, out.cut(i * 2 * k, (i + 1) * 2 * k));
                           knnHelper2<dim, nodeT, objT>(tree, queries[i], buf);
                           buf.keepK();
                           if (sorted)
                             buf.sort();
                           for (size_t j = 0; j < k; ++j)
                           {
                             idx[i * k + j] = buf[j].entry - queries.begin();
                           }
                         });
    if (freeTree)
      free(tree);
    return idx;
  }

  template <int dim, typename nodeT, typename objT>
  void knnRangeHelper(nodeT *tree, objT &q, point<dim> qMin, point<dim> qMax,
                      double radius, knnBuf::buffer<objT *> &out)
  {
    int relation = tree->boxCompare(qMin, qMax, tree->getMin(), tree->getMax());

    if (relation == tree->boxExclude)
    {
      return;
    }
    else if (relation == tree->boxInclude)
    {
      for (size_t i = 0; i < tree->size(); ++i)
      {
        objT *p = tree->getItem(i);
        out.insert(knnBuf::elem(q.dist(*p), p));
      }
    }
    else
    { // intersect
      if (tree->isLeaf())
      {
        for (size_t i = 0; i < tree->size(); ++i)
        {
          objT *p = tree->getItem(i);
          double dist = q.dist(*p);
          if (dist <= radius)
          {
            out.insert(knnBuf::elem(dist, p));
          }
        }
      }
      else
      {
        knnRangeHelper<dim, nodeT, objT>(tree->L(), q, qMin, qMax, radius, out);
        knnRangeHelper<dim, nodeT, objT>(tree->R(), q, qMin, qMax, radius, out);
      }
    }
  }

  template <int dim, typename nodeT, typename objT>
  void knnRange(nodeT *tree, objT &q, double radius, knnBuf::buffer<objT *> &out)
  {
    point<dim> qMin, qMax;
    for (size_t i = 0; i < dim; i++)
    {
      auto tmp = q[i] - radius;
      qMin[i] = tmp;
      qMax[i] = tmp + radius * 2;
    }
    knnRangeHelper<dim, nodeT, objT>(tree, q, qMin, qMax, radius, out);
  }

  template <int dim, typename nodeT, typename objT>
  void knnHelper(nodeT *tree, objT &q, knnBuf::buffer<objT *> &out)
  {
    // find the leaf first
    int relation = tree->boxCompare(tree->getMin(), tree->getMax(),
                                    point<dim>(q.coords()),
                                    point<dim>(q.coords()));
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
          out.insert(knnBuf::elem(q.dist(*p), p));
        }
      }
      else
      {
        knnHelper<dim, nodeT, objT>(tree->L(), q, out);
        knnHelper<dim, nodeT, objT>(tree->R(), q, out);
      }
    }

    if (!out.hasK())
    {
      if (tree->siblin() == NULL)
      {
        throw std::runtime_error("Error, knnHelper reached root node without enough neighbors.");
      }
      for (size_t i = 0; i < tree->siblin()->size(); ++i)
      {
        objT *p = tree->siblin()->getItem(i);
        out.insert(knnBuf::elem(q.dist(*p), p));
      }
    }
    else
    { // Buffer filled to a least k
      if (tree->siblin() != NULL)
      {
        knnBuf::elem tmp = out.keepK();
        knnRange<dim, nodeT, objT>(tree->siblin(), q, tmp.cost, out);
      }
    }
  }

  template <int dim, class objT>
  parlay::sequence<size_t> batchKnn(parlay::slice<objT *, objT *> queries,
                                    size_t k,
                                    node<dim, objT> *tree,
                                    bool sorted)
  {
    using nodeT = node<dim, objT>;
    bool freeTree = false;
    if (!tree)
    {
      freeTree = true;
      tree = build<dim, objT>(queries, true);
    }
    auto out = parlay::sequence<knnBuf::elem<objT *>>(2 * k * queries.size());
    auto idx = parlay::sequence<size_t>(k * queries.size());
    parlay::parallel_for(0, queries.size(), [&](size_t i)
                         {
                           knnBuf::buffer buf = knnBuf::buffer<objT *>(k, out.cut(i * 2 * k, (i + 1) * 2 * k));
                           knnHelper<dim, nodeT, objT>(tree, queries[i], buf);
                           buf.keepK();
                           if (sorted)
                             buf.sort();
                           for (size_t j = 0; j < k; ++j)
                           {
                             idx[i * k + j] = buf[j].entry - queries.begin();
                           }
                         });
    if (freeTree)
      free(tree);
    return idx;
  }

  template <int dim, typename nodeT, typename objT>
  void traverseTree(nodeT *tree, objT &query, knnBuf::buffer<objT *> &out, int k)
  {
    if (tree->isLeaf())
    {
      // basecase
      for (size_t i = 0; i < tree->size(); ++i)
      {
        objT *p = tree->getItem(i);
        out.insert(knnBuf::elem(query.dist(*p), p));
      }
      return;
    }
    traverseTree<dim, nodeT, objT>(tree->left, query, out, k);
    traverseTree<dim, nodeT, objT>(tree->right, query, out, k);
  }

  template <int dim, class objT>
  parlay::sequence<size_t> batchTraverse(parlay::slice<objT *, objT *> queries,
                                    size_t k,
                                    node<dim, objT> *tree = nullptr,
                                    bool sorted = false)
  {
    using nodeT = node<dim, objT>;
    bool freeTree = false;
    if (!tree)
    {
      freeTree = true;
      tree = build<dim, objT>(queries, true);
    }
    auto out = parlay::sequence<knnBuf::elem<objT *>>(2 * k * queries.size());
    auto idx = parlay::sequence<size_t>(k * queries.size());
    parlay::parallel_for(0, queries.size(), [&](size_t i)
                         {
                           knnBuf::buffer buf = knnBuf::buffer<objT *>(k, out.cut(i * 2 * k, (i + 1) * 2 * k));
                           traverseTree<dim, nodeT, objT>(tree, queries[i], buf, k);
                           buf.keepK();
                           if (sorted)
                             buf.sort();
                           for (size_t j = 0; j < k; ++j)
                           {
                             idx[i * k + j] = buf[j].entry - queries.begin();
                           }
                         });
    if (freeTree)
      free(tree);
    return idx;
  }

  //   template <int dim, typename nodeT, typename objT>
  // void knnHelperSimple(nodeT *tree, objT &query, knnBuf::buffer<objT *> &out, int k)
  // {

  //   if (tree->isLeaf())
  //   {
  //     // basecase
  //     for (size_t i = 0; i < tree->size(); ++i)
  //     {
  //       objT *p = tree->getItem(i);
  //       out.insert(knnBuf::elem(query.dist(*p), p));
  //     }
  //     return;
  //   }

  //   const double split = tree->getSplit();
  //   const int axis = tree->k;

  //   nodeT* next[2] = {tree->left, tree->right}; //next[0] = tree->left; next[1] = tree->right;

  //   const int dir = query[axis] < split ? 0 : 1;
  //   knnHelperSimple<dim, nodeT, objT>(next[dir], query, out, k);
  //   // nnSearchRecursive(query, node->next[dir], guess, minDist);

	// 	const double diff = fabs(query[axis] - split);
  //   if ((int)out.size() < k || diff < out.back()) knnHelperSimple<dim, nodeT, objT>(next[!dir], query, out, k);
  //     // knnSearchRecursive(query, node->next[!dir], queue, k);
  // }

  // template <int dim, class objT>
  // parlay::sequence<size_t> batchKnnSimple(parlay::slice<objT *, objT *> queries,
  //                                   size_t k,
  //                                   node<dim, objT> *tree = nullptr,
  //                                   bool sorted = false)
  // {
  //   using nodeT = node<dim, objT>;
  //   bool freeTree = false;
  //   if (!tree)
  //   {
  //     freeTree = true;
  //     tree = build<dim, objT>(queries, true);
  //   }
  //   auto out = parlay::sequence<knnBuf::elem<objT *>>(2 * k * queries.size());
  //   auto idx = parlay::sequence<size_t>(k * queries.size());
  //   parlay::parallel_for(0, queries.size(), [&](size_t i)
  //                        {
  //                          knnBuf::buffer buf = knnBuf::buffer<objT *>(k, out.cut(i * 2 * k, (i + 1) * 2 * k));
  //                          knnHelperSimple<dim, nodeT, objT>(tree, queries[i], buf, k);
  //                         //  std::cout <<"done" << std::endl;
  //                          buf.keepK();
  //                          if (sorted)
  //                            buf.sort();
  //                          for (size_t j = 0; j < k; ++j)
  //                          {
  //                            idx[i * k + j] = buf[j].entry - queries.begin();
  //                          }
  //                         //   if(i==0){
  //                         //    for (size_t j = 0; j < k; ++j)
  //                         //  {
  //                         //    std::cout <<  buf[j].entry - queries.begin() << " " << buf[j].cost << std::endl;
  //                         //  }
  //                         //  }
  //                        });
  //   if (freeTree)
  //     free(tree);
  //   return idx;
  // }
  // template <int dim, class objT>
  // parlay::sequence<size_t> batchKnnOmp(parlay::slice<objT *, objT *> queries,
  //                                   size_t k,
  //                                   node<dim, objT> *tree,
  //                                   bool sorted)
  // {
  //   using nodeT = node<dim, objT>;
  //   bool freeTree = false;
  //   if (!tree)
  //   {
  //     freeTree = true;
  //     tree = build<dim, objT>(queries, true);
  //   }
  //   auto out = parlay::sequence<knnBuf::elem<objT *>>(2 * k * queries.size());
  //   auto idx = parlay::sequence<size_t>(k * queries.size());
  //   #pragma omp parallel for // num_threads( numThreads/2 )  proc_bind(master)
  //   for ( size_t i = 0; i < queries.size(); i++ ){
  //         // int place_num = omp_get_place_num();
  //         knnBuf::buffer buf = knnBuf::buffer<objT *>(k, out.cut(i * 2 * k, (i + 1) * 2 * k));
  //         knnHelper<dim, nodeT, objT>(tree, queries[i], buf);
  //         buf.keepK();
  //         if (sorted)
  //           buf.sort();
  //         for (size_t j = 0; j < k; ++j)
  //         {
  //           idx[i * k + j] = buf[j].entry - queries.begin();
  //         }          
  //   }
  //   if (freeTree)
  //     free(tree);
  //   return idx;
  // }

  template <int dim, typename objT>
  parlay::sequence<size_t> bruteforceKnn(parlay::sequence<objT> &queries, size_t k)
  {
    auto out = parlay::sequence<knnBuf::elem<objT *>>(2 * k * queries.size());
    auto idx = parlay::sequence<size_t>(k * queries.size());
    parlay::parallel_for(0, queries.size(), [&](size_t i)
                         {
                           objT q = queries[i];
                           knnBuf::buffer buf = knnBuf::buffer<objT *>(k, out.cut(i * 2 * k, (i + 1) * 2 * k));
                           for (size_t j = 0; j < queries.size(); ++j)
                           {
                             objT *p = &queries[j];
                             buf.insert(elem(q.dist(p), p));
                           }
                           buf.keepK();
                           for (size_t j = 0; j < k; ++j)
                           {
                             idx[i * k + j] = buf[j].entry - queries.data();
                           }
                         });
    return idx;
  }

} // End namespace pargeo
