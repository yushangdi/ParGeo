#include "parlay/hash_table.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "common/geometry.h"
#include "incremental.h"
#include "hull.h"

parlay::sequence<facet3d<pargeo::fpoint<3>>> hull3d(parlay::sequence<pargeo::fpoint<3>> &P, size_t numProc) {
  using namespace std;
  using namespace parlay;
  using floatT = pargeo::fpoint<3>::floatT;

  size_t n = P.size();
  cout << "#-points = " << n << endl;
  cout << "#-procs = " << num_workers() << endl;

  auto H = incrementHull3d<pargeo::fpoint<3>>(make_slice(P), numProc);
  cout << H.size() << endl;

  return H;
}

parlay::sequence<facet3d<pargeo::fpoint<3>>> hull3d(parlay::sequence<pargeo::fpoint<3>> &, size_t);
