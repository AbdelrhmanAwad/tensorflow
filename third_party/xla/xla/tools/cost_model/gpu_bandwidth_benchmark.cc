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

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace xla::gpu {
namespace {

constexpr absl::string_view kTableComment =
    "// Lookup table entry mapping DMA transfer size to fractional saturation\n"
    "// of peak bandwidth in [0.0, 1.0].\n";

}  // namespace

std::string FormatCppTable(absl::Span<const BandwidthEntry> entries,
                           absl::string_view variable_name) {
  std::string result(kTableComment);
  absl::StrAppendFormat(&result,
                        "constexpr std::array<BandwidthEntry, %v> %s = {\n",
                        entries.size(), variable_name);

  for (const BandwidthEntry& entry : entries) {
    absl::StrAppendFormat(&result, "    {%v, %.8ff},\n", entry.dma_size_bytes,
                          entry.bandwidth_fraction);
  }

  absl::StrAppend(&result, "};\n");
  return result;
}

}  // namespace xla::gpu
