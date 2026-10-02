#include "arbitrium/data.h"

#include <cstdlib>
#include <iostream>

void check_data(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

int main() {
  try {
    std::vector<arbitrium::EncodedSample> samples{
        {"b", "f2", {4, 9, 3}, {0, 2, 3}, -1, false},
        {"a", "f1", {4, 10, 11, 3}, {0, 2, 2, 3}, 1, true},
        {"c", "f3", {4, 12, 3}, {0, 2, 3}, 2, true},
    };
    for (int index = 3; index < 12; ++index) {
      samples.push_back({"extra" + std::to_string(index), "fx",
                         {4, 9 + index, 3}, {0, 2, 3}, 0, true});
    }
    const auto first = arbitrium::epoch_order(samples.size(), 42, 0);
    const auto again = arbitrium::epoch_order(samples.size(), 42, 0);
    const auto next = arbitrium::epoch_order(samples.size(), 42, 1);
    check_data(first == again && first != next, "stable epoch shuffle");
    const std::vector<size_t> ordered{0, 1, 2};
    const auto batch = arbitrium::make_batch(samples, ordered, 0, 2);
    check_data(batch.token_ids.sizes() == torch::IntArrayRef({2, 4}), "right padded dimensions");
    check_data(batch.attention_mask[0].sum().item<int64_t>() == 3 &&
                   batch.attention_mask[1].sum().item<int64_t>() == 4,
               "mask lengths");
    check_data(batch.targets[0].item<int64_t>() == -1 &&
                   !batch.answerable[0].item<bool>(),
               "unanswerable target retained only outside model input");
    check_data(batch.token_ids[0][3].item<int64_t>() == 0 &&
                   batch.segment_ids[0][3].item<int64_t>() == 0,
               "padding contract");
    std::cout << "data batching passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
