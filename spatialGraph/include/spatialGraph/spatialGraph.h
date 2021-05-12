#pragma once

#include "parlay/sequence.h"
#include "pargeo/edge.h"
#include "pargeo/point.h"

namespace pargeo {

using edge = pargeo::_edge<false, size_t>;
using dirEdge = pargeo::_edge<true, size_t>;

template<int dim>
parlay::sequence<pargeo::dirEdge> knnGraph(parlay::sequence<pargeo::point<dim>> &S, size_t k);

template<int dim>
parlay::sequence<pargeo::edge> delaunayGraph(parlay::sequence<pargeo::point<dim>> &S);

template<int dim>
parlay::sequence<pargeo::edge> gabrielGraph(parlay::sequence<pargeo::point<dim>> &S);

template<int dim>
parlay::sequence<pargeo::edge> betaSkeleton(parlay::sequence<pargeo::point<dim>> &S, double);

}