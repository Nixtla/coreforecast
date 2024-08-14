# cython: language_level=3
# distutils: language=c++
from libc.stdint cimport int32_t


cdef extern from "grouped_array.h":
    ctypedef int32_t indptr_t

    cdef cppclass GroupedArray[T]:
        GroupedArray(const T* data, const indptr_t*, int, int)
        void LagTransform(int, T*)

        void IndexFromEnd(int, T*)
        void Head(int, T*)
        void Tail(int, T*)
        void Tails(const indptr_t*, T*)
        void Append(const GroupedArray&, const indptr_t*, T*)

        void RollingMeanTransform(int, int, int, T*)
        void RollingStdTransform(int, int, int, T*)
        void RollingMinTransform(int, int, int, T*)
        void RollingMaxTransform(int, int, int, T*)
        void RollingQuantileTransform(int, T, int, int, T*)
        void RollingMeanUpdate(int, int, int, T*)
        void RollingStdUpdate(int, int, int, T*)
        void RollingMinUpdate(int, int, int, T*)
        void RollingMaxUpdate(int, int, int, T*)
        void RollingQuantileUpdate(int, T, int, int, T*)

        void SeasonalRollingMeanTransform(int, int, int, int, T*)
        void SeasonalRollingStdTransform(int, int, int, int, T*)
        void SeasonalRollingMinTransform(int, int, int, int, T*)
        void SeasonalRollingMaxTransform(int, int, int, int, T*)
        void SeasonalRollingQuantileTransform(int, T, int, int, int, T*)
        void SeasonalRollingMeanUpdate(int, int, int, int, T*)
        void SeasonalRollingStdUpdate(int, int, int, int, T*)
        void SeasonalRollingMinUpdate(int, int, int, int, T*)
        void SeasonalRollingMaxUpdate(int, int, int, int, T*)
        void SeasonalRollingQuantileUpdate(int, T, int, int, int, T*)

        void ExpandingMeanTransform(int, T*, T*)
        void ExpandingStdTransform(int, T*, T*)
        void ExpandingMinTransform(int, T*)
        void ExpandingMaxTransform(int, T*)
        void ExpandingQuantileTransform(int, T, T*)
        void ExpandingQuantileUpdate(int, T, T*)

        void ExponentiallyWeightedMeanTransform(int, T, T*)

        void MinMaxScalerStats(T*)
        void StandardScalerStats(T*)
        void RobustIqrScalerStats(T*)
        void RobustMadScalerStats(T*)
        void ApplyScaler(const T*, T*)
        void InvertScaler(const T*, T*)
        void BoxCoxLambdaGuerrero(int, T, T, T*)
        void BoxCoxLambdaLogLik(int, T, T, T*)
        void BoxCoxTransform(const T*, T*)
        void BoxCoxInverseTransform(const T*, T*)

        void NumDiffs(int, T*)
        void NumSeasDiffs(int, int, T*)
        void NumSeasDiffsPeriods(int, T*)
        void Period(size_t, T*)
        void Difference(int, T*)
        void Differences(const indptr_t*, T*)
        void InvertDifferences(const GroupedArray&, const indptr_t*, T*)
