//
//  demo_module.cpp
//  python-multi-array
//
//  Copyright (C) 2017 Rue Yokaze
//  Distributed under the MIT License.
//
#include <boost/multi_array.hpp>
#include <boost/python.hpp>

using boost::multi_array;
using std::shared_ptr;

float average(const shared_ptr<multi_array<float, 1>>& vec)
{
    float total = 0.f;
    size_t size = vec->shape()[0];
    for (size_t i = 0; i < size; ++i)
    {
        total += (*vec)[i];
    }
    return total / size;
}

BOOST_PYTHON_MODULE(demo_module)
{
    using boost::python::def;
    def("average", average);
}
