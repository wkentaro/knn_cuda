// Python
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/numeric.hpp>
#include <numpy/ndarrayobject.h>
#include "knn.h"

using namespace boost::python;

// For extracting features from a 4-D blob feature map.
object extract_feature(PyObject* activation_, PyObject* coords_)
{
  PyArrayObject* activation_py = (PyArrayObject*) activation_;
  PyArrayObject* coords_py     = (PyArrayObject*) coords_;
  int n_batch   = PyArray_DIM(activation_py, 0);
  int n_channel = PyArray_DIM(activation_py, 1);
  int height    = PyArray_DIM(activation_py, 2);
  int width     = PyArray_DIM(activation_py, 3);

  int n_max_coord = PyArray_DIM(coords_py, 1);
  int dim_coord   = PyArray_DIM(coords_py, 2);

  float* activation           = new float[n_batch * n_channel * height * width];
  float* coords               = new float[n_batch * n_max_coord * dim_coord];
  float* extracted_activation = new float[n_batch * n_channel * n_max_coord];;

  // Copy python objects
  for(int n = 0; n < n_batch; n++){
    for (int c = 0; c < n_channel; c++){
      for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
          activation[((n * n_channel + c) * height + i) * width + j] =
              *(float*)PyArray_GETPTR4(activation_py, n, c, i, j);
        }
      }
    }
  }

  for(int n = 0; n < n_batch; n++){
    for(int i = 0; i < n_max_coord; i++) {
      for(int j = 0; j < dim_coord; j++) {
        coords[(n * n_max_coord + i) * dim_coord + j] =
            *(float*)PyArray_GETPTR3(coords_py, n, i, j);
      }
    }
  }

  extract_cuda(activation, n_batch, n_channel, height,
      width, coords, n_max_coord, dim_coord, extracted_activation);

  npy_intp dims[3] = {n_batch, n_channel, n_max_coord};
  PyObject* py_obj = PyArray_SimpleNewFromData(3, dims, NPY_FLOAT,
                                               extracted_activation);
  handle<> handle(py_obj);

  boost::python::object arr(handle);

  free(activation);
  free(coords);

  return arr;
}

// CUDA K-NN wrapper
// Takes features and retuns the distances and indices of the k-nearest
// neighboring features.
object knn(PyObject* query_points_, PyObject* ref_points_, int k)
{
  PyArrayObject* query_points = (PyArrayObject*) query_points_;
  PyArrayObject* ref_points   = (PyArrayObject*) ref_points_;
  int n_query = PyArray_DIM(query_points, 1);
  int n_ref   = PyArray_DIM(ref_points, 1);
  int dim     = PyArray_DIM(query_points, 0);
  float* query_points_c = new float[n_query * dim];
  float* ref_points_c   = new float[n_ref * dim];
  float* dist = new float[n_query * k];
  int* ind    = new int[n_query * k];

  // Copy python objects
  for(int i = 0; i < n_query; i++) {
    for(int j = 0; j < dim; j++) {
      query_points_c[n_query * j + i] =
          *(float*)PyArray_GETPTR2(query_points, j, i);
    }
  }

  for(int i = 0; i < n_ref; i++) {
    for(int j = 0; j < dim; j++) {
      ref_points_c[n_ref * j + i] = *(float*)PyArray_GETPTR2(ref_points, j, i);
    }
  }

  knn_cuda(ref_points_c, n_ref, query_points_c, n_query, dim, k, dist, ind);

  npy_intp dims[2] = {k, n_query};
  PyObject* py_obj_dist = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, dist);
  PyObject* py_obj_ind  = PyArray_SimpleNewFromData(2, dims, NPY_INT, ind);
  handle<> handle_dist(py_obj_dist);
  handle<> handle_ind(py_obj_ind);

  boost::python::object arr_dist(handle_dist);
  boost::python::object arr_ind(handle_ind);

  free(query_points_c);
  free(ref_points_c);

  return make_tuple(arr_dist, arr_ind);
}

int init_numpy() {
  import_array();
}

BOOST_PYTHON_MODULE(knn)
{
  init_numpy();
  def("knn", knn);
  def("extract", extract_feature);
}
