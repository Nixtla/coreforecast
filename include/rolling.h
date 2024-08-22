#pragma once

#include "SkipList.h"

#include "stats.h"

namespace rolling
{
  template <typename T>
  inline void MeanTransform(const T *data, int n, T *out, int window_size,
                            int min_samples)
  {
    T accum = static_cast<T>(0.0);
    int upper_limit = std::min(window_size, n);
    for (int i = 0; i < upper_limit; ++i)
    {
      accum += data[i];
      if (i + 1 < min_samples)
      {
        out[i] = std::numeric_limits<T>::quiet_NaN();
      }
      else
      {
        out[i] = accum / (i + 1);
      }
    }

    for (int i = window_size; i < n; ++i)
    {
      accum += data[i] - data[i - window_size];
      out[i] = accum / window_size;
    }
  }

  template <typename T>
  inline void StdTransformWithStats(const T *data, int n, T *out, T *agg,
                                    bool save_stats, int window_size, int min_samples)
  {
    T prev_avg = static_cast<T>(0.0);
    T curr_avg = data[0];
    T m2 = static_cast<T>(0.0);
    int upper_limit = std::min(window_size, n);
    for (int i = 0; i < upper_limit; ++i)
    {
      prev_avg = curr_avg;
      curr_avg = prev_avg + (data[i] - prev_avg) / (i + 1);
      m2 += (data[i] - prev_avg) * (data[i] - curr_avg);
      if (i + 1 < min_samples)
      {
        out[i] = std::numeric_limits<T>::quiet_NaN();
      }
      else
      {
        out[i] = sqrt(m2 / i);
      }
    }
    for (int i = window_size; i < n; ++i)
    {
      T delta = data[i] - data[i - window_size];
      prev_avg = curr_avg;
      curr_avg = prev_avg + delta / window_size;
      m2 += delta * (data[i] - curr_avg + data[i - window_size] - prev_avg);
      // avoid possible loss of precision
      m2 = std::max(m2, static_cast<T>(0.0));
      out[i] = sqrt(m2 / (window_size - 1));
    }
    if (save_stats)
    {
      agg[0] = static_cast<T>(n);
      agg[1] = curr_avg;
      agg[2] = m2;
    }
  }

  template <typename T>
  inline void StdTransform(const T *data, int n, T *out, int window_size,
                           int min_samples)
  {
    T tmp;
    StdTransformWithStats(data, n, out, &tmp, false, window_size, min_samples);
  }

  template <typename T, typename Comp>
  class SortedDeque
  {
  public:
    SortedDeque(int window_size, Comp comp = Comp())
        : window_size_(window_size), comp_(comp)
    {
      buffer_.reserve(window_size);
    }
    inline bool empty() const noexcept { return tail_ == -1; }
    inline void push_back(int i, T x) noexcept
    {
      if (tail_ == -1)
      {
        head_ = 0;
        tail_ = 0;
      }
      else if (tail_ == window_size_ - 1)
      {
        tail_ = 0;
      }
      else
      {
        ++tail_;
      }
      buffer_[tail_] = {i, x};
    }
    inline void pop_back() noexcept
    {
      if (head_ == tail_)
      {
        head_ = 0;
        tail_ = -1;
      }
      else if (tail_ == 0)
      {
        tail_ = window_size_ - 1;
      }
      else
      {
        --tail_;
      }
    }
    inline void pop_front() noexcept
    {
      if (head_ == tail_)
      {
        head_ = 0;
        tail_ = -1;
      }
      else if (head_ == window_size_ - 1)
      {
        head_ = 0;
      }
      else
      {
        ++head_;
      }
    }
    inline const std::pair<int, T> &front() const noexcept
    {
      return buffer_[head_];
    }
    inline const std::pair<int, T> &back() const noexcept
    {
      return buffer_[tail_];
    }
    void update(T x) noexcept
    {
      while (!empty() && comp_(back().second, x))
      {
        pop_back();
      }
      if (!empty() && front().first <= i_)
      {
        pop_front();
      }
      push_back(window_size_ + i_, x);
      ++i_;
    }
    T get() const noexcept { return front().second; }

  private:
    std::vector<std::pair<int, T>> buffer_;
    int window_size_;
    int head_ = 0;
    int tail_ = -1;
    int i_ = 0;
    Comp comp_;
  };

  template <typename T, typename Comp>
  inline void CompTransform(const T *data, int n, T *out, int window_size,
                            int min_samples)
  {
    int upper_limit = std::min(window_size, n);
    SortedDeque<T, Comp> sdeque(window_size);
    for (int i = 0; i < upper_limit; ++i)
    {
      sdeque.update(data[i]);
      if (i + 1 < min_samples)
      {
        out[i] = std::numeric_limits<T>::quiet_NaN();
      }
      else
      {
        out[i] = sdeque.get();
      }
    }
    for (int i = upper_limit; i < n; ++i)
    {
      sdeque.update(data[i]);
      out[i] = sdeque.get();
    }
  }

  template <typename T>
  void MinTransform(const T *data, int n, T *out, int window_size,
                    int min_samples)
  {
    CompTransform<T, std::greater_equal<T>>(data, n, out, window_size,
                                            min_samples);
  }

  template <typename T>
  void MaxTransform(const T *data, int n, T *out, int window_size,
                    int min_samples)
  {
    CompTransform<T, std::less_equal<T>>(data, n, out, window_size, min_samples);
  }

  template <typename T>
  inline void QuantileTransform(const T *data, int n, T *out, int window_size,
                                int min_samples, T p)
  {
    int upper_limit = std::min(window_size, n);
    OrderedStructs::SkipList::HeadNode<T> sl;
    for (int i = 0; i < upper_limit; ++i)
    {
      sl.insert(data[i]);
      if (i + 1 < min_samples)
      {
        out[i] = std::numeric_limits<T>::quiet_NaN();
      }
      else
      {
        out[i] = SortedQuantile(sl, p, i + 1);
      }
    }
    for (int i = window_size; i < n; ++i)
    {
      sl.remove(data[i - window_size]);
      sl.insert(data[i]);
      out[i] = SortedQuantile(sl, p, window_size);
    }
  }

  template <typename Func, typename T, typename... Args>
  inline void SeasonalTransform(Func RollingTfm, const T *data, int n, T *out,
                                int season_length, int window_size,
                                int min_samples, Args &&...args)
  {
    int buff_size = n / season_length + (n % season_length > 0);
    T *season_data = new T[buff_size];
    T *season_out = new T[buff_size];
    std::fill_n(season_out, buff_size, std::numeric_limits<T>::quiet_NaN());
    for (int i = 0; i < season_length; ++i)
    {
      int season_n = n / season_length + (i < n % season_length);
      for (int j = 0; j < season_n; ++j)
      {
        season_data[j] = data[i + j * season_length];
      }
      RollingTfm(season_data, season_n, season_out, window_size, min_samples,
                 std::forward<Args>(args)...);
      for (int j = 0; j < season_n; ++j)
      {
        out[i + j * season_length] = season_out[j];
      }
    }
    delete[] season_data;
    delete[] season_out;
  }

  template <typename T>
  struct SeasonalMeanTransform
  {
    void operator()(const T *data, int n, T *out, int season_length,
                    int window_size, int min_samples)
    {
      SeasonalTransform(MeanTransform<T>, data, n, out, season_length,
                        window_size, min_samples);
    }
  };

  template <typename T>
  struct SeasonalStdTransform
  {
    void operator()(const T *data, int n, T *out, int season_length,
                    int window_size, int min_samples)
    {
      SeasonalTransform(StdTransform<T>, data, n, out, season_length, window_size,
                        min_samples);
    }
  };

  template <typename T>
  struct SeasonalMinTransform
  {
    void operator()(const T *data, int n, T *out, int season_length,
                    int window_size, int min_samples)
    {
      SeasonalTransform(MinTransform<T>, data, n, out, season_length, window_size,
                        min_samples);
    }
  };

  template <typename T>
  struct SeasonalMaxTransform
  {
    void operator()(const T *data, int n, T *out, int season_length,
                    int window_size, int min_samples)
    {
      SeasonalTransform(MaxTransform<T>, data, n, out, season_length, window_size,
                        min_samples);
    }
  };

  template <typename T>
  struct SeasonalQuantileTransform
  {
    void operator()(const T *data, int n, T *out, int season_length,
                    int window_size, int min_samples, T p)
    {
      SeasonalTransform(QuantileTransform<T>, data, n, out, season_length,
                        window_size, min_samples, p);
    }
  };

  template <typename Func, typename T, typename... Args>
  inline void Update(Func RollingTfm, const T *data, int n, T *out,
                     int window_size, int min_samples, Args &&...args)
  {
    if (n < min_samples)
    {
      *out = std::numeric_limits<T>::quiet_NaN();
      return;
    }
    int n_samples = std::min(window_size, n);
    T *buffer = new T[n_samples];
    RollingTfm(data + n - n_samples, n_samples, buffer, window_size, min_samples,
               std::forward<Args>(args)...);
    *out = buffer[n_samples - 1];
    delete[] buffer;
  }

  template <typename T>
  struct MeanUpdate
  {
    void operator()(const T *data, int n, T *out, int window_size,
                    int min_samples)
    {
      Update(MeanTransform<T>, data, n, out, window_size, min_samples);
    }
  };

  template <typename T>
  struct StdUpdate
  {
    void operator()(const T *data, int n, T *out, int window_size,
                    int min_samples)
    {
      Update(StdTransform<T>, data, n, out, window_size, min_samples);
    }
  };

  template <typename T>
  struct MinUpdate
  {
    void operator()(const T *data, int n, T *out, int window_size,
                    int min_samples)
    {
      Update(MinTransform<T>, data, n, out, window_size, min_samples);
    }
  };

  template <typename T>
  struct MaxUpdate
  {
    void operator()(const T *data, int n, T *out, int window_size,
                    int min_samples)
    {
      Update(MaxTransform<T>, data, n, out, window_size, min_samples);
    }
  };

  template <typename T>
  struct QuantileUpdate
  {
    void operator()(const T *data, int n, T *out, int window_size,
                    int min_samples, T p)
    {
      Update(QuantileTransform<T>, data, n, out, window_size, min_samples, p);
    }
  };

  template <typename Func, typename T, typename... Args>
  inline void SeasonalUpdate(Func RollingUpdate, const T *data, int n, T *out,
                             int season_length, int window_size, int min_samples,
                             Args &&...args)
  {
    int season = n % season_length;
    int season_n = n / season_length + (season > 0);
    if (season_n < min_samples)
    {
      *out = std::numeric_limits<T>::quiet_NaN();
      return;
    }
    int n_samples = std::min(window_size, season_n);
    T *season_data = new T[n_samples];
    for (int i = 0; i < n_samples; ++i)
    {
      season_data[i] = data[n - 1 - (n_samples - 1 - i) * season_length];
    }
    RollingUpdate(season_data, n_samples, out, window_size, min_samples,
                  std::forward<Args>(args)...);
    delete[] season_data;
  }

  template <typename T>
  struct SeasonalMeanUpdate
  {
    void operator()(const T *data, int n, T *out, int season_length,
                    int window_size, int min_samples)
    {
      SeasonalUpdate(MeanUpdate<T>(), data, n, out, season_length, window_size,
                     min_samples);
    }
  };

  template <typename T>
  struct SeasonalStdUpdate
  {
    void operator()(const T *data, int n, T *out, int season_length,
                    int window_size, int min_samples)
    {
      SeasonalUpdate(StdUpdate<T>(), data, n, out, season_length, window_size,
                     min_samples);
    }
  };

  template <typename T>
  struct SeasonalMinUpdate
  {
    void operator()(const T *data, int n, T *out, int season_length,
                    int window_size, int min_samples)
    {
      SeasonalUpdate(MinUpdate<T>(), data, n, out, season_length, window_size,
                     min_samples);
    }
  };

  template <typename T>
  struct SeasonalMaxUpdate
  {
    void operator()(const T *data, int n, T *out, int season_length,
                    int window_size, int min_samples)
    {
      SeasonalUpdate(MaxUpdate<T>(), data, n, out, season_length, window_size,
                     min_samples);
    }
  };

  template <typename T>
  struct SeasonalQuantileUpdate
  {
    void operator()(const T *data, int n, T *out, int season_length,
                    int window_size, int min_samples, T p)
    {
      SeasonalUpdate(QuantileUpdate<T>(), data, n, out, season_length,
                     window_size, min_samples, p);
    }
  };
} // namespace rolling
