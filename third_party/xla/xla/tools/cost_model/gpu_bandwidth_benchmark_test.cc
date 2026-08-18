/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/tools/cost_model/gpu_bandwidth_benchmark.h"

#include <vector>

#include <gtest/gtest.h>

namespace xla::gpu {
namespace {

TEST(GpuBandwidthBenchmarkTest, FormatCppTableEmpty) {
  EXPECT_EQ(
      FormatCppTable({}, "kEmptyTable"),
      R"(// Lookup table entry mapping DMA transfer size to fractional saturation
// of peak bandwidth in [0.0, 1.0].
constexpr std::array<BandwidthEntry, 0> kEmptyTable = {
};
)");
}

TEST(GpuBandwidthBenchmarkTest, FormatCppTableMultipleEntries) {
  std::vector<BandwidthEntry> entries = {
      {8192, 0.00043418f},
      {16384, 0.00092645f},
      {32768, 0.00184066f},
      {65536, 0.00359877f},
  };
  EXPECT_EQ(
      FormatCppTable(entries, "kFourTable"),
      R"(// Lookup table entry mapping DMA transfer size to fractional saturation
// of peak bandwidth in [0.0, 1.0].
constexpr std::array<BandwidthEntry, 4> kFourTable = {
    {8192, 0.00043418f},
    {16384, 0.00092645f},
    {32768, 0.00184066f},
    {65536, 0.00359877f},
};
)");
}

TEST(GpuBandwidthBenchmarkTest, FormatCppTableLargeDmaSizes) {
  std::vector<BandwidthEntry> entries = {
      {2147483648LL, 0.5f},
      {4294967296LL, 0.75f},
      {8589934592LL, 1.0f},
  };
  EXPECT_EQ(
      FormatCppTable(entries, "kLargeTable"),
      R"(// Lookup table entry mapping DMA transfer size to fractional saturation
// of peak bandwidth in [0.0, 1.0].
constexpr std::array<BandwidthEntry, 3> kLargeTable = {
    {2147483648, 0.50000000f},
    {4294967296, 0.75000000f},
    {8589934592, 1.00000000f},
};
)");
}

}  // namespace
}  // namespace xla::gpu
