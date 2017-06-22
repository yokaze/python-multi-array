//
//  python_multi_array.cpp
//  python-multi-array
//
//  Copyright (C) 2017 Rue Yokaze
//  Distributed under the MIT License.
//
#include <boost/multi_array.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <memory>
#include <stdint.h>

//  Using from python is avoided because many definitions conflict with names from std.
namespace python = boost::python;
using boost::extents;
using boost::multi_array;
using std::shared_ptr;

static python::object numpy = python::import("numpy");
static python::object bool8 = numpy.attr("bool8");
static python::object uint8 = numpy.attr("uint8");
static python::object uint16 = numpy.attr("uint16");
static python::object uint32 = numpy.attr("uint32");
static python::object uint64 = numpy.attr("uint64");
static python::object int8 = numpy.attr("int8");
static python::object int16 = numpy.attr("int16");
static python::object int32 = numpy.attr("int32");
static python::object int64 = numpy.attr("int64");
static python::object float32 = numpy.attr("float32");
static python::object float64 = numpy.attr("float64");

namespace python_multi_array
{
    //
    //  [Python]
    //  [array_type] multi_array.make(shape, dtype)
    //
    //  allocate a boost::multi_array of expected shape and data type.
    //
    //  shape: int, list or tuple
    //  dtype: bool8, int8, int16, int32, int64, uint8, uint16, uint32, uint64,
    //         float32 or float64, all defined in numpy
    //
    //  return: a smart-pointer of the array
    //
    python::object make(python::object shape, python::object dtype);

    namespace impl
    {
        template <class T>
        python::object make_typed_sized(const size_t* s, size_t ndim)
        {
            switch (ndim)
            {
            case 1:
                return python::object(std::make_shared<multi_array<T, 1>>(extents[s[0]]));
            case 2:
                return python::object(std::make_shared<multi_array<T, 2>>(extents[s[0]][s[1]]));
            case 3:
                return python::object(std::make_shared<multi_array<T, 3>>(extents[s[0]][s[1]][s[2]]));
            case 4:
                return python::object(std::make_shared<multi_array<T, 4>>(extents[s[0]][s[1]][s[2]][s[3]]));
            case 5:
                return python::object(std::make_shared<multi_array<T, 5>>(extents[s[0]][s[1]][s[2]][s[3]][s[4]]));
            case 6:
                return python::object(std::make_shared<multi_array<T, 6>>(extents[s[0]][s[1]][s[2]][s[3]][s[4]][s[5]]));
            case 7:
                return python::object(std::make_shared<multi_array<T, 7>>(extents[s[0]][s[1]][s[2]][s[3]][s[4]][s[5]][s[6]]));
            case 8:
                return python::object(std::make_shared<multi_array<T, 8>>(extents[s[0]][s[1]][s[2]][s[3]][s[4]][s[5]][s[6]][s[7]]));
            default:
                throw std::invalid_argument("shape");
            }
        }

        template <class T>
        python::object make_typed(python::object shape)
        {
            python::extract<size_t> scalar_shape(shape);
            if (scalar_shape.check())
            {
                size_t shape = static_cast<size_t>(scalar_shape);
                return make_typed_sized<T>(&shape, 1);
            }
            size_t ndim = python::len(shape);
            size_t s[ndim];
            for (size_t i = 0; i < ndim; ++i)
            {
                s[i] = python::extract<size_t>(shape[i]);
            }
            return make_typed_sized<T>(s, ndim);
        }
    }

    python::object make(python::object shape, python::object dtype)
    {
        if (dtype == bool8)
        {
            return impl::make_typed<bool>(shape);
        }
        else if (dtype == int8)
        {
            return impl::make_typed<int8_t>(shape);
        }
        else if (dtype == int16)
        {
            return impl::make_typed<int16_t>(shape);
        }
        else if (dtype == int32)
        {
            return impl::make_typed<int32_t>(shape);
        }
        else if (dtype == int64)
        {
            return impl::make_typed<int16_t>(shape);
        }
        else if (dtype == uint8)
        {
            return impl::make_typed<uint8_t>(shape);
        }
        else if (dtype == uint16)
        {
            return impl::make_typed<uint16_t>(shape);
        }
        else if (dtype == uint32)
        {
            return impl::make_typed<uint32_t>(shape);
        }
        else if (dtype == uint64)
        {
            return impl::make_typed<uint64_t>(shape);
        }
        else if (dtype == float32)
        {
            return impl::make_typed<float>(shape);
        }
        else if (dtype == float64)
        {
            return impl::make_typed<double>(shape);
        }
        else
        {
            throw std::invalid_argument("dtype");
        }
    }

    //
    //  [Python]
    //  T x[idx]
    //  x[idx] = T
    //
    //  get and set one element via index operator.
    //  Example:
    //    x[2, 4] = 2.0
    //
    template <class T, size_t N>
    T getitem(const shared_ptr<multi_array<T, N>>& This, python::object idx);

    template <class T, size_t N>
    void setitem(const shared_ptr<multi_array<T, N>>& This, python::object idx, T value);

    namespace impl
    {
        template <class T, size_t N>
        T getitem_impl(const shared_ptr<multi_array<T, N>>& This, const size_t* s)
        {
            T* ptr = This->origin();
            for (size_t i = 0; i < N; ++i)
            {
                if (This->shape()[i] <= s[i])
                {
                    throw std::invalid_argument("index");
                }
                ptr += This->strides()[i] * s[i];
            }
            return *ptr;
        }

        template <class T, size_t N>
        void setitem_impl(const shared_ptr<multi_array<T, N>>& This, const size_t* s, T value)
        {
            T* ptr = This->origin();
            for (size_t i = 0; i < N; ++i)
            {
                if (This->shape()[i] <= s[i])
                {
                    throw std::invalid_argument("index");
                }
                ptr += This->strides()[i] * s[i];
            }
            *ptr = value;
        }
    }

    template <class T, size_t N>
    T getitem(const shared_ptr<multi_array<T, N>>& This, python::object idx)
    {
        if (This == nullptr)
        {
            throw std::invalid_argument("self");
        }
        if (N == 1)
        {
            // if N = 1, an integer value can be used for indexing
            python::extract<size_t> scalar_idx(idx);
            if (scalar_idx.check())
            {
                size_t index = static_cast<size_t>(scalar_idx);
                return impl::getitem_impl(This, &index);
            }
        }
        //  assume idx to be a list or a tuple
        if (N != python::len(idx))
        {
            throw std::invalid_argument("index");
        }
        size_t s[N];
        for (size_t i = 0; i < N; ++i)
        {
            s[i] = python::extract<size_t>(idx[i]);
        }
        return impl::getitem_impl(This, s);
    }

    template <class T, size_t N>
    void setitem(const shared_ptr<multi_array<T, N>>& This, python::object idx, T value)
    {
        if (This == nullptr)
        {
            throw std::invalid_argument("self");
        }
        if (N == 1)
        {
            // if N = 1, an integer value can be used for indexing
            python::extract<size_t> scalar_idx(idx);
            if (scalar_idx.check())
            {
                size_t index = static_cast<size_t>(scalar_idx);
                impl::setitem_impl(This, &index, value);
                return;
            }
        }
        //  assume idx to be a list or a tuple
        if (N != python::len(idx))
        {
            throw std::invalid_argument("index");
        }
        size_t s[N];
        for (size_t i = 0; i < N; ++i)
        {
            s[i] = python::extract<size_t>(idx[i]);
        }
        impl::setitem_impl(This, s, value);
    }

    //
    //  [Python]
    //  x.reset()
    //
    //  This function resets every elements of the array with zero.
    //
    template <class T, size_t N>
    void reset(const shared_ptr<multi_array<T, N>>& This)
    {
        if (This == nullptr)
        {
            throw std::invalid_argument("self");
        }
        std::fill(This->origin(), This->origin() + This->num_elements(), 0);
    }

    //
    //  [Python]
    //  dtype x.element()
    //
    //  return: data type of the array. possible values are bool8, uint8,
    //          uint16, uint32, uint64, int8, int16, int32, int64, float32,
    //          float64, all defined in numpy.
    //
    template <class T, size_t N>
    python::object element(const shared_ptr<multi_array<T, N>>& This)
    {
        if (This == nullptr)
        {
            throw std::invalid_argument("self");
        }
        return python::numpy::dtype::get_builtin<T>();
    }

    //
    //  [Python]
    //  tuple x.shape()
    //
    //  return: the shape of the array.
    //
    template <class T, size_t N>
    python::object shape(const shared_ptr<multi_array<T, N>>& This)
    {
        if (This == nullptr)
        {
            throw std::invalid_argument("self");
        }
        const size_t* s = This->shape();
        switch (N)
        {
        case 1:
            return python::make_tuple(s[0]);
        case 2:
            return python::make_tuple(s[0], s[1]);
        case 3:
            return python::make_tuple(s[0], s[1], s[2]);
        case 4:
            return python::make_tuple(s[0], s[1], s[2], s[3]);
        case 5:
            return python::make_tuple(s[0], s[1], s[2], s[3], s[4]);
        case 6:
            return python::make_tuple(s[0], s[1], s[2], s[3], s[4], s[5]);
        case 7:
            return python::make_tuple(s[0], s[1], s[2], s[3], s[4], s[5], s[6]);
        case 8:
            return python::make_tuple(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]);
        default:
            throw std::invalid_argument("self");
        }
    }

    //
    //  [Python]
    //  int x.num_dimensions()
    //
    //  return: the number of dimensions of the array.
    //
    template <class T, size_t N>
    size_t num_dimensions(const shared_ptr<multi_array<T, N>>& This)
    {
        if (This == nullptr)
        {
            throw std::invalid_argument("self");
        }
        return N;
    }

    //
    //  [Python]
    //  int x.num_elements()
    //
    //  return: the total number of elements of the array.
    //  example:
    //    It returns 8 for an array with shape (2, 4).
    //
    template <class T, size_t N>
    size_t num_elements(const shared_ptr<multi_array<T, N>>& This)
    {
        if (This == nullptr)
        {
            throw std::invalid_argument("self");
        }
        return This->num_elements();
    }

    //
    //  [Python]
    //  numpy.ndarray x.get()
    //
    //  return: a copy of the array stored in numpy.ndarray.
    //
    template <class T, size_t N>
    python::object get(const shared_ptr<multi_array<T, N>>& This)
    {
        if (This == nullptr)
        {
            throw std::invalid_argument("self");
        }
        size_t s[N];
        size_t d[N];
        std::copy(This->shape(), This->shape() + N, s);
        std::transform(This->strides(), This->strides() + N, d, [](auto input) { return input * sizeof(T); });
        auto make_tuple_from_array = [](const size_t* a) {
            switch (N)
            {
            case 1:
                return python::make_tuple(a[0]);
            case 2:
                return python::make_tuple(a[0], a[1]);
            case 3:
                return python::make_tuple(a[0], a[1], a[2]);
            case 4:
                return python::make_tuple(a[0], a[1], a[2], a[3]);
            case 5:
                return python::make_tuple(a[0], a[1], a[2], a[3], a[4]);
            case 6:
                return python::make_tuple(a[0], a[1], a[2], a[3], a[4], a[5]);
            case 7:
                return python::make_tuple(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
            case 8:
                return python::make_tuple(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
            }
            throw std::invalid_argument("this");
        };
        python::numpy::dtype dt = python::numpy::dtype::get_builtin<T>();
        python::tuple shape = make_tuple_from_array(s);
        python::tuple strides = make_tuple_from_array(d);
        return boost::python::numpy::from_data(This->origin(), dt, shape, strides, boost::python::object());
    }

    //
    //  [Python]
    //  x.set(numpy.ndarray nd)
    //
    //  Reset the array with values from nd.
    //  nd.dtype may be different from x.element() but the values are implicitly
    //  converted to x.element().
    //
    template <class T, size_t N>
    void set(const shared_ptr<multi_array<T, N>>& This, python::numpy::ndarray nd);

    namespace impl
    {
        template <class T, size_t N, class S>
        void set_typed(shared_ptr<multi_array<T, N>> This, python::numpy::ndarray nd)
        {
            if (N != nd.get_nd())
            {
                throw std::invalid_argument("nd");
            }

            size_t s[8];
            std::fill(std::begin(s), std::end(s), 0);
            std::copy(This->shape(), This->shape() + N, s);

            size_t dt[8];
            std::fill(std::begin(dt), std::end(dt), 0);
            std::copy(This->strides(), This->strides() + N, dt);

            size_t dnd[8];
            std::fill(std::begin(dnd), std::end(dnd), 0);
            std::transform(nd.get_strides(), nd.get_strides() + N, dnd, [](auto input) { return input / sizeof(S); });

            for (size_t i = 0; i < N; ++i)
            {
                if (nd.get_shape()[i] == 1)
                {
                    dnd[i] = 0;
                }
                else if (s[i] != nd.get_shape()[i])
                {
                    throw std::invalid_argument("nd");
                }
            }

            T* pt = This->origin();
            const S* pnd = reinterpret_cast<S*>(nd.get_data());

            for (size_t i0 = 0; i0 < s[0]; ++i0)
            {
                T* pt1 = pt + i0 * dt[0];
                const S* pnd1 = pnd + i0 * dnd[0];
                *pt1 = static_cast<T>(*pnd1);
                for (size_t i1 = 1; i1 < s[1]; ++i1)
                {
                    T* pt2 = pt1 + i1 * dt[1];
                    const S* pnd2 = pnd1 + i1 * dnd[1];
                    *pt2 = static_cast<T>(*pnd2);
                    for (size_t i2 = 1; i2 < s[2]; ++i2)
                    {
                        T* pt3 = pt2 + i2 * dt[2];
                        const S* pnd3 = pnd2 + i2 * dnd[2];
                        *pt3 = static_cast<T>(*pnd3);
                        for (size_t i3 = 1; i3 < s[3]; ++i3)
                        {
                            T* pt4 = pt3 + i3 * dt[3];
                            const S* pnd4 = pnd3 + i3 * dnd[3];
                            *pt4 = static_cast<T>(*pnd4);
                            for (size_t i4 = 1; i4 < s[4]; ++i4)
                            {
                                T* pt5 = pt4 + i4 * dt[4];
                                const S* pnd5 = pnd4 + i4 * dnd[4];
                                *pt5 = static_cast<T>(*pnd5);
                                for (size_t i5 = 1; i5 < s[5]; ++i5)
                                {
                                    T* pt6 = pt5 + i5 * dt[5];
                                    const S* pnd6 = pnd5 + i5 * dnd[5];
                                    *pt6 = static_cast<T>(*pnd6);
                                    for (size_t i6 = 1; i6 < s[6]; ++i6)
                                    {
                                        T* pt7 = pt6 + i6 * dt[6];
                                        const S* pnd7 = pnd6 + i6 * dnd[6];
                                        *pt7 = static_cast<T>(*pnd7);
                                        for (size_t i7 = 1; i7 < s[7]; ++i7)
                                        {
                                            T* pt8 = pt7 + i7 * dt[7];
                                            const S* pnd8 = pnd7 + i7 * dnd[7];
                                            *pt8 = static_cast<T>(*pnd8);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    template <class T, size_t N>
    void set(const shared_ptr<multi_array<T, N>>& This, python::numpy::ndarray nd)
    {
        if (This == nullptr)
        {
            throw std::invalid_argument("self");
        }
        python::numpy::dtype dt = nd.get_dtype();
        if (dt == bool8)
        {
            impl::set_typed<T, N, bool>(This, nd);
        }
        else if (dt == uint8)
        {
            impl::set_typed<T, N, uint8_t>(This, nd);
        }
        else if (dt == uint16)
        {
            impl::set_typed<T, N, uint16_t>(This, nd);
        }
        else if (dt == uint32)
        {
            impl::set_typed<T, N, uint32_t>(This, nd);
        }
        else if (dt == uint64)
        {
            impl::set_typed<T, N, uint64_t>(This, nd);
        }
        else if (dt == int8)
        {
            impl::set_typed<T, N, int8_t>(This, nd);
        }
        else if (dt == int16)
        {
            impl::set_typed<T, N, int16_t>(This, nd);
        }
        else if (dt == int32)
        {
            impl::set_typed<T, N, int32_t>(This, nd);
        }
        else if (dt == int64)
        {
            impl::set_typed<T, N, int64_t>(This, nd);
        }
        else if (dt == float32)
        {
            impl::set_typed<T, N, float>(This, nd);
        }
        else if (dt == float64)
        {
            impl::set_typed<T, N, double>(This, nd);
        }
        else
        {
            throw std::invalid_argument("nd");
        }
    }

    //
    //  [Internal-usage only]
    //  let python interpreter to export types from this module.
    //
    class array_template
    {
    public:
        template <class T, size_t N>
        static void declare(const char* name)
        {
            python::class_<shared_ptr<multi_array<T, N>>>(name)
                .def("__getitem__", &getitem<T, N>)
                .def("__setitem__", &setitem<T, N>)
                .def("reset", &reset<T, N>)
                .def("element", &element<T, N>)
                .def("shape", &shape<T, N>)
                .def("num_dimensions", &num_dimensions<T, N>)
                .def("num_elements", &num_elements<T, N>)
                .def("get", &get<T, N>)
                .def("set", &set<T, N>);
        }
    };
}

BOOST_PYTHON_MODULE(multi_array)
{
    using namespace python_multi_array;
    using boost::python::def;
    boost::python::numpy::initialize();

    array_template::declare<bool, 1>("shared_bool_vector");
    array_template::declare<bool, 2>("shared_bool_matrix");
    array_template::declare<bool, 3>("shared_bool_tensor");
    array_template::declare<bool, 4>("shared_bool_tensor4");
    array_template::declare<bool, 5>("shared_bool_tensor5");
    array_template::declare<bool, 6>("shared_bool_tensor6");
    array_template::declare<bool, 7>("shared_bool_tensor7");
    array_template::declare<bool, 8>("shared_bool_tensor8");
    array_template::declare<uint8_t, 1>("shared_uint8_vector");
    array_template::declare<uint8_t, 2>("shared_uint8_matrix");
    array_template::declare<uint8_t, 3>("shared_uint8_tensor");
    array_template::declare<uint8_t, 4>("shared_uint8_tensor4");
    array_template::declare<uint8_t, 5>("shared_uint8_tensor5");
    array_template::declare<uint8_t, 6>("shared_uint8_tensor6");
    array_template::declare<uint8_t, 7>("shared_uint8_tensor7");
    array_template::declare<uint8_t, 8>("shared_uint8_tensor8");
    array_template::declare<uint16_t, 1>("shared_uint16_vector");
    array_template::declare<uint16_t, 2>("shared_uint16_matrix");
    array_template::declare<uint16_t, 3>("shared_uint16_tensor");
    array_template::declare<uint16_t, 4>("shared_uint16_tensor4");
    array_template::declare<uint16_t, 5>("shared_uint16_tensor5");
    array_template::declare<uint16_t, 6>("shared_uint16_tensor6");
    array_template::declare<uint16_t, 7>("shared_uint16_tensor7");
    array_template::declare<uint16_t, 8>("shared_uint16_tensor8");
    array_template::declare<uint32_t, 1>("shared_uint32_vector");
    array_template::declare<uint32_t, 2>("shared_uint32_matrix");
    array_template::declare<uint32_t, 3>("shared_uint32_tensor");
    array_template::declare<uint32_t, 4>("shared_uint32_tensor4");
    array_template::declare<uint32_t, 5>("shared_uint32_tensor5");
    array_template::declare<uint32_t, 6>("shared_uint32_tensor6");
    array_template::declare<uint32_t, 7>("shared_uint32_tensor7");
    array_template::declare<uint32_t, 8>("shared_uint32_tensor8");
    array_template::declare<uint64_t, 1>("shared_uint64_vector");
    array_template::declare<uint64_t, 2>("shared_uint64_matrix");
    array_template::declare<uint64_t, 3>("shared_uint64_tensor");
    array_template::declare<uint64_t, 4>("shared_uint64_tensor4");
    array_template::declare<uint64_t, 5>("shared_uint64_tensor5");
    array_template::declare<uint64_t, 6>("shared_uint64_tensor6");
    array_template::declare<uint64_t, 7>("shared_uint64_tensor7");
    array_template::declare<uint64_t, 8>("shared_uint64_tensor8");
    array_template::declare<int8_t, 1>("shared_int8_vector");
    array_template::declare<int8_t, 2>("shared_int8_matrix");
    array_template::declare<int8_t, 3>("shared_int8_tensor");
    array_template::declare<int8_t, 4>("shared_int8_tensor4");
    array_template::declare<int8_t, 5>("shared_int8_tensor5");
    array_template::declare<int8_t, 6>("shared_int8_tensor6");
    array_template::declare<int8_t, 7>("shared_int8_tensor7");
    array_template::declare<int8_t, 8>("shared_int8_tensor8");
    array_template::declare<int16_t, 1>("shared_int16_vector");
    array_template::declare<int16_t, 2>("shared_int16_matrix");
    array_template::declare<int16_t, 3>("shared_int16_tensor");
    array_template::declare<int16_t, 4>("shared_int16_tensor4");
    array_template::declare<int16_t, 5>("shared_int16_tensor5");
    array_template::declare<int16_t, 6>("shared_int16_tensor6");
    array_template::declare<int16_t, 7>("shared_int16_tensor7");
    array_template::declare<int16_t, 8>("shared_int16_tensor8");
    array_template::declare<int32_t, 1>("shared_int32_vector");
    array_template::declare<int32_t, 2>("shared_int32_matrix");
    array_template::declare<int32_t, 3>("shared_int32_tensor");
    array_template::declare<int32_t, 4>("shared_int32_tensor4");
    array_template::declare<int32_t, 5>("shared_int32_tensor5");
    array_template::declare<int32_t, 6>("shared_int32_tensor6");
    array_template::declare<int32_t, 7>("shared_int32_tensor7");
    array_template::declare<int32_t, 8>("shared_int32_tensor8");
    array_template::declare<int64_t, 1>("shared_int64_vector");
    array_template::declare<int64_t, 2>("shared_int64_matrix");
    array_template::declare<int64_t, 3>("shared_int64_tensor");
    array_template::declare<int64_t, 4>("shared_int64_tensor4");
    array_template::declare<int64_t, 5>("shared_int64_tensor5");
    array_template::declare<int64_t, 6>("shared_int64_tensor6");
    array_template::declare<int64_t, 7>("shared_int64_tensor7");
    array_template::declare<int64_t, 8>("shared_int64_tensor8");
    array_template::declare<float, 1>("shared_float_vector");
    array_template::declare<float, 2>("shared_float_matrix");
    array_template::declare<float, 3>("shared_float_tensor");
    array_template::declare<float, 4>("shared_float_tensor4");
    array_template::declare<float, 5>("shared_float_tensor5");
    array_template::declare<float, 6>("shared_float_tensor6");
    array_template::declare<float, 7>("shared_float_tensor7");
    array_template::declare<float, 8>("shared_float_tensor8");
    array_template::declare<double, 1>("shared_double_vector");
    array_template::declare<double, 2>("shared_double_matrix");
    array_template::declare<double, 3>("shared_double_tensor");
    array_template::declare<double, 4>("shared_double_tensor4");
    array_template::declare<double, 5>("shared_double_tensor5");
    array_template::declare<double, 6>("shared_double_tensor6");
    array_template::declare<double, 7>("shared_double_tensor7");
    array_template::declare<double, 8>("shared_double_tensor8");

    def("make", make);

    //  define aliases of numpy data types
    python::scope This;
    This.attr("bool8") = bool8;
    This.attr("uint8") = uint8;
    This.attr("uint16") = uint16;
    This.attr("uint32") = uint32;
    This.attr("uint64") = uint64;
    This.attr("int8") = int8;
    This.attr("int16") = int16;
    This.attr("int32") = int32;
    This.attr("int64") = int64;
    This.attr("float32") = float32;
    This.attr("float64") = float64;
}
