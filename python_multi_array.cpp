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
#include <vector>

//  Using from python is avoided because many definitions conflict with names from std.
namespace python = boost::python;
using boost::extents;
using boost::multi_array;
using std::shared_ptr;
using std::vector;

static python::object main_module = python::import("__main__");
static python::object builtin_module = main_module.attr("__builtins__");
static python::object python_hasattr = builtin_module.attr("hasattr");
static python::object python_int = builtin_module.attr("int");

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
    namespace impl
    {
        size_t extract_size_t(python::object obj)
        {
            //  numpy.int32 cannot be converted directly into C++ integer types.
            //  therefore, the value is first converted to python integer type
            //  and then converted to size_t.
            return python::extract<size_t>(python_int(obj));
        }

        vector<size_t> extract_index(python::object index)
        {
            if (python::extract<bool>(python_hasattr(index, "__len__")) == false)
            {
                return { extract_size_t(index) };
            }
            else
            {
                vector<size_t> ret;
                size_t length = python::len(index);
                for (size_t i = 0; i < length; ++i)
                {
                    ret.push_back(extract_size_t(index[i]));
                }
                return ret;
            }
        }
    }

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
            auto reset = [](const auto& arr)
            {
                std::fill(arr->origin(), arr->origin() + arr->num_elements(), 0);
                return arr;
            };

            switch (ndim)
            {
                case 1:
                    return python::object(reset(std::make_shared<multi_array<T, 1>>(extents[s[0]])));
                case 2:
                    return python::object(reset(std::make_shared<multi_array<T, 2>>(extents[s[0]][s[1]])));
                case 3:
                    return python::object(reset(std::make_shared<multi_array<T, 3>>(extents[s[0]][s[1]][s[2]])));
                case 4:
                    return python::object(reset(std::make_shared<multi_array<T, 4>>(extents[s[0]][s[1]][s[2]][s[3]])));
                case 5:
                    return python::object(reset(std::make_shared<multi_array<T, 5>>(extents[s[0]][s[1]][s[2]][s[3]][s[4]])));
                case 6:
                    return python::object(reset(std::make_shared<multi_array<T, 6>>(extents[s[0]][s[1]][s[2]][s[3]][s[4]][s[5]])));
                case 7:
                    return python::object(reset(std::make_shared<multi_array<T, 7>>(extents[s[0]][s[1]][s[2]][s[3]][s[4]][s[5]][s[6]])));
                case 8:
                    return python::object(reset(std::make_shared<multi_array<T, 8>>(extents[s[0]][s[1]][s[2]][s[3]][s[4]][s[5]][s[6]][s[7]])));
                default:
                    throw std::invalid_argument("shape");
            }
        }

        template <class T>
        python::object make_typed(python::object shape)
        {
            vector<size_t> shape_vector = extract_index(shape);
            return make_typed_sized<T>(shape_vector.data(), shape_vector.size());
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
            return impl::make_typed<int64_t>(shape);
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
    //  T x[index]
    //  x[index] = T
    //
    //  get and set one element via index operator.
    //  Example:
    //    x[2, 4] = 2.0
    //
    template <class T, size_t N>
    T getitem(const shared_ptr<multi_array<T, N>>& This, python::object index_object)
    {
        if (This == nullptr)
        {
            throw std::invalid_argument("self");
        }
        vector<size_t> index = impl::extract_index(index_object);
        if (index.size() != N)
        {
            //  index has an invalid dimensionality
            throw std::invalid_argument("index");
        }
        T* ptr = This->origin();
        for (size_t i = 0; i < N; ++i)
        {
            if (This->shape()[i] <= index[i])
            {
                //  index exceeds boundary
                throw std::invalid_argument("index");
            }
            ptr += This->strides()[i] * index[i];
        }
        return *ptr;
    }

    template <class T, size_t N>
    void setitem(const shared_ptr<multi_array<T, N>>& This, python::object index_object, T value)
    {
        if (This == nullptr)
        {
            throw std::invalid_argument("self");
        }
        vector<size_t> index = impl::extract_index(index_object);
        if (index.size() != N)
        {
            //  index has an invalid dimensionality
            throw std::invalid_argument("index");
        }
        T* ptr = This->origin();
        for (size_t i = 0; i < N; ++i)
        {
            if (This->shape()[i] <= index[i])
            {
                throw std::invalid_argument("index");
            }
            ptr += This->strides()[i] * index[i];
        }
        *ptr = value;
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
            size_t s[N];
            std::copy(This->shape(), This->shape() + N, s);
            size_t ix[N];
            std::fill(ix, ix + N, 0);

            size_t boost_strides[N];
            std::copy(This->strides(), This->strides() + N, boost_strides);

            size_t numpy_strides[N];
            std::transform(nd.get_strides(), nd.get_strides() + N, numpy_strides, [](auto input) { return input / sizeof(S); });

            T* p_boost_origin = This->origin();
            const S* p_numpy_origin = reinterpret_cast<S*>(nd.get_data());

            while (ix[0] < s[0])
            {
                T* p_boost_element = p_boost_origin;
                const S* p_numpy_element = p_numpy_origin;
                for (size_t d = 0; d < (N - 1); ++d)
                {
                    p_boost_element += ix[d] * boost_strides[d];
                    p_numpy_element += ix[d] * numpy_strides[d];
                }
                while (ix[N - 1] < s[N - 1])
                {
                    *p_boost_element = static_cast<T>(*p_numpy_element);
                    p_boost_element += boost_strides[N - 1];
                    p_numpy_element += numpy_strides[N - 1];
                    ++(ix[N - 1]);
                }
                for (size_t d = N - 1; d > 0; --d)
                {
                    if (s[d] <= ix[d])
                    {
                        ix[d] = 0;
                        ++(ix[d - 1]);
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
