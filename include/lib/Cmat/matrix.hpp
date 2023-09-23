#pragma once
#include <memory>
#include <vector>
#include <assert.h>
#include <thread>
#include <iostream>
# include "../thread-pool/BS_thread_pool.hpp"

namespace Cmat
{

struct Matrix 
{
    int m_width, m_height;
    std::vector<std::vector<double>>m_data;
    size_t n_cpu;
    std::shared_ptr<BS::thread_pool> pool;

    Matrix():
        m_width(0), 
        m_height(0),
        n_cpu(std::thread::hardware_concurrency()),
        pool(std::make_shared<BS::thread_pool>(n_cpu))
    {}
    ~Matrix() {}

    Matrix(std::vector<double>v):
        n_cpu(std::thread::hardware_concurrency()),
        pool(std::make_shared<BS::thread_pool>(n_cpu))
    {
        m_data = {v};
        m_width = v.size();
        m_height = 1;
    }

    Matrix(std::vector<std::vector<double>>v):
        n_cpu(std::thread::hardware_concurrency()),
        pool(std::make_shared<BS::thread_pool>(n_cpu))
    {
        m_data = v;
        m_width = v.size()?v[0].size():0;
        m_height = v.size();
    }

    Matrix(int height, int width):
        n_cpu(std::thread::hardware_concurrency()),
        pool(std::make_shared<BS::thread_pool>(n_cpu))
    {
        *this = Matrix::zeros(height, width);
    }

    Matrix zeros(size_t height, size_t width) const {
        return Matrix{ std::vector<std::vector<double>>(height, std::vector<double>(width, 0)) };
    }

    void push_back(std::vector<double>row)
    {
        assert((int)row.size() == m_width or m_height==0);

        m_data.push_back(row);

        if (m_height == 0) m_width = row.size();
        m_height++;
    }

    void print()
    {
        for (int i=0; i<m_height; ++i)
        {
            for (int j=0; j<m_width; ++j)
            {
                std::cout << m_data[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    Matrix operator+(const Matrix m) const { return Matrix(*this) += m; }
    Matrix operator-(const Matrix m) const { return Matrix(*this) -= m; }
    Matrix operator*(const Matrix m) const { return Matrix(*this) *= m; }
    Matrix operator*(const double a) const { return Matrix(*this) *= a; }
    Matrix operator/(const Matrix m) const { return Matrix(*this) /= m; }

    Matrix operator+=(const Matrix m)
    {
        assert(m_width!=m.m_width and m_height!=m.m_height);

        for (int i = 0; i < m_height; ++i)
        {
            pool->push_task([&](int i){
                for (int j = 0; j < m_width; ++j)
                {
                    m_data[i][j] = m_data[i][j] + m.m_data[i][j];
                }
            }, i);
        }
        pool->wait_for_tasks();

        return *this;
    }

    Matrix operator-=(const Matrix m)
    {
        for (int i = 0; i < m_height; ++i)
        {
            pool->push_task([&](int i){
                for (int j = 0; j < m_width; ++j)
                {
                    m_data[i][j] = m_data[i][j] - m.m_data[i][j];
                }
            }, i);
        }
        pool->wait_for_tasks();

        return *this;
    }

    Matrix operator*=(const double a)
    {
        for (int i = 0; i < m_height; ++i)
        {
            pool->push_task([&](int i){
                for (int j = 0; j < m_width; ++j)
                {
                    m_data[i][j] = m_data[i][j] * a;
                }
            }, i);
        }
        pool->wait_for_tasks();

        return *this;
    }

    Matrix operator*=(const Matrix m)
    {
        for (int i = 0; i < m_height; ++i)
        {
            pool->push_task([&](int i){
                for (int j = 0; j < m_width; ++j)
                {
                    m_data[i][j] = m_data[i][j] * m.m_data[i][j];
                }
            }, i);
        }
        pool->wait_for_tasks();

        return *this;
    }

    Matrix operator/=(const Matrix m)
    {
        for (int i = 0; i < m_height; ++i)
        {
            pool->push_task([&](int i){
                for (int j = 0; j < m_width; ++j)
                {
                    m_data[i][j] = m_data[i][j] / m.m_data[i][j];
                }
            }, i);
        }
        pool->wait_for_tasks();

        return *this;
    }

    std::vector<double>& operator[](int i)
    {
        return m_data[i];
    }

    Matrix dot(const Matrix m)
    {
        assert(m_width == m.m_height);

        Matrix mat = Matrix::zeros(m_height, m.m_width);

        for (int idx = 0; idx < m_height; ++idx)
        {
            pool->push_task([&](int idx){
                for (int i=0; i<m_width; ++i)
                {
                    for (int j=0; j<m.m_width; ++j)
                    {
                        mat.m_data[idx][j] += m_data[idx][i] * m.m_data[i][j];
                    }
                }
            }, idx);
        }

        pool->wait_for_tasks();

        return mat;
    }

    Matrix transpose() const
    {
        Matrix mat{ m_width, m_height };

        for (int i = 0; i < m_height; ++i)
        {
            pool->push_task([&](int i){
                for (int j = 0; j < m_width; ++j)
                {
                    mat.m_data[j][i] = m_data[i][j];
                }
            }, i);
        }
        pool->wait_for_tasks();

        return mat;
    }

    double sum() const
    {
        double res = 0;

        for (int i = 0; i < m_height; ++i)
        {
            pool->push_task([&](int i){
                for (int j = 0; j < m_width; ++j)
                {
                    res += m_data[i][j];
                }
            }, i);
        }

        pool->wait_for_tasks();

        return res;
    }
};

}