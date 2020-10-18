// Copyright (c) 2020 Yiqiu Wang and the Pargeo Team
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

#ifndef BRIO_H
#define BRIO_H

#include "spatialSort.h"

template<int dim, class T>
void brioSortSerial(T* A, intT n, intT prefix=1000) {
  if (n < prefix) return;
  intT x = n/2;
  spatialSortSerial<dim, T>(A+x, n-x);
  brioSortSerial<dim, T>(A, x, prefix);
}

template<int dim, class T>
void brioSort(T* A, intT n, intT prefix=1000) {
  if (n < prefix) return;
  if (n < 2000) return brioSortSerial<dim, T>(A, n, prefix);
  intT x = n/2;
  cilk_spawn spatialSort<dim, T>(A+x, n-x);
  brioSort<dim, T>(A, x, prefix);
  cilk_sync;
}

#endif