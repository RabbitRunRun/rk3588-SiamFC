#ifndef _INC_CANN_H
#define _INC_CANN_H

#include <iostream>
#include <string>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <memory>
#include <algorithm>
#include <numeric>

//#include "acl/acl.h"

//#include "ohm/need.h"

#include "seeta_aip.h"

class ACLTensor {
public:
    using self = ACLTensor;

    ACLTensor(const ACLTensor &) = delete;
    
    ACLTensor &operator=(const ACLTensor &) = delete;

    ACLTensor(ACLTensor &&that) {
        std::swap(m_dataType, that.m_dataType);
        std::swap(m_dims, that.m_dims);
        std::swap(m_host_data, that.m_host_data);
        std::swap(m_count, that.m_count);
    }
    
    ACLTensor &operator=(ACLTensor &&that) {
        std::swap(m_dataType, that.m_dataType);
        std::swap(m_dims, that.m_dims);
        std::swap(m_host_data, that.m_host_data);
        std::swap(m_count, that.m_count);
        return *this;
    }

    ACLTensor()
            : m_dataType() {}

    ACLTensor(const std::vector<int64_t> &dims)
            : m_dataType(SEETA_AIP_VALUE_FLOAT32)
            , m_dims(dims) {
        m_count = std::accumulate(m_dims.begin(), m_dims.end(), 1, std::multiplies<int64_t>());
       
        m_host_data = new float[m_count * sizeof(float)]; 
        //auto err = aclrtMallocHost(&m_host_data, m_count * sizeof(float));
        //if (err != ACL_SUCCESS) throw ACLException("aclrtMallocHost", err);
    }

    ~ACLTensor() {
        if (m_host_data) {
            //aclrtFreeHost(m_host_data);
            delete [] m_host_data;
            m_host_data = nullptr;
        }
    }

    float *data() { return (float *)m_host_data; }
    
    const float *data() const { return (float *)m_host_data; }

    SEETA_AIP_VALUE_TYPE type() const { return m_dataType; }

    const std::vector<int64_t> &dims() const { return m_dims; }
    
    const std::vector<int64_t> &shape() const { return m_dims; }

    template <typename I, typename = typename std::enable_if<std::is_integral<I>::value>::type>
    int64_t dim(I i) const { return m_dims[i]; }
    
    template <typename I, typename = typename std::enable_if<std::is_integral<I>::value>::type>
    int64_t shape(I i) const { return m_dims[i]; }

    int64_t count() const { return m_count; }

    int64_t bytes() const { return m_count * sizeof(float); }

    template <typename I, typename = typename std::enable_if<std::is_integral<I>::value>::type>
    float &at(I i) { return data()[i]; }
    
    template <typename I, typename = typename std::enable_if<std::is_integral<I>::value>::type>
    const float &at(I i) const { return data()[i]; }
    
    template <typename I0, typename I1,
        typename = typename std::enable_if<
            std::is_integral<I0>::value && 
            std::is_integral<I1>::value 
            >::type>
    float &at(I0 i0, I1 i1) { return data()[i0 * m_dims[1] + i1]; }
    
    template <typename I0, typename I1,
        typename = typename std::enable_if<
            std::is_integral<I0>::value && 
            std::is_integral<I1>::value 
            >::type>
    const float &at(I0 i0, I1 i1) const { return data()[i0 * m_dims[1] + i1]; }
    
    template <typename I0, typename I1, typename I2,
        typename = typename std::enable_if<
            std::is_integral<I0>::value && 
            std::is_integral<I1>::value && 
            std::is_integral<I2>::value 
            >::type>
    float &at(I0 i0, I1 i1, I2 i2) { return data()[(i0 * m_dims[1] + i1) * m_dims[2] + i2]; }
    
    template <typename I0, typename I1, typename I2,
        typename = typename std::enable_if<
            std::is_integral<I0>::value && 
            std::is_integral<I1>::value && 
            std::is_integral<I2>::value 
            >::type>
    const float &at(I0 i0, I1 i1, I2 i2) const { return data()[(i0 * m_dims[1] + i1) * m_dims[2] + i2]; }
    
    template <typename I0, typename I1, typename I2, typename I3,
        typename = typename std::enable_if<
            std::is_integral<I0>::value && 
            std::is_integral<I1>::value && 
            std::is_integral<I2>::value && 
            std::is_integral<I3>::value 
            >::type>
    float &at(I0 i0, I1 i1, I2 i2, I3 i3) { return data()[((i0 * m_dims[1] + i1) * m_dims[2] + i2) * m_dims[3] + i3]; }
    
    template <typename I0, typename I1, typename I2, typename I3,
        typename = typename std::enable_if<
            std::is_integral<I0>::value && 
            std::is_integral<I1>::value && 
            std::is_integral<I2>::value && 
            std::is_integral<I3>::value 
            >::type>
    const float &at(I0 i0, I1 i1, I2 i2, I3 i3) const { return data()[((i0 * m_dims[1] + i1) * m_dims[2] + i2) * m_dims[3] + i3]; }
private:
    // assume that using float only
    //aclDataType m_dataType = ACL_FLOAT;

    SEETA_AIP_VALUE_TYPE m_dataType = SEETA_AIP_VALUE_FLOAT32;
    std::vector<int64_t> m_dims;
    void *m_host_data = nullptr;

    int64_t m_count = 0;
};


#endif // _INC_CANN_H

