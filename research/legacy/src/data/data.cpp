#include "arbitrium/data.h"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <set>

namespace arbitrium {

std::vector<EncodedSample> load_split(const std::string& path, const std::string& split,
                                      const std::string& expected_sha256,
                                      const TaskSpec& task, const Tokenizer& tokenizer) {
  const auto bytes = read_file(path, 1024ULL * 1024 * 1024);
  if (sha256(bytes) != expected_sha256) throw ContractError("split hash mismatch", "artifact_invalid");
  std::vector<EncodedSample> result;
  std::set<std::string> ids;
  size_t start = 0;
  while (start < bytes.size()) {
    const auto end = bytes.find('\n', start);
    if (end == std::string::npos) throw ContractError("JSONL must end with LF");
    const auto length = end - start;
    if (length == 0) throw ContractError("blank JSONL line");
    if (length > 128 * 1024) throw ContractError("sample line too long", "input_too_long");
    auto sample = Sample::from_json(parse_json(bytes.substr(start, length), 128 * 1024), task);
    if (sample.split != split) throw ContractError("split record mismatch");
    if (!ids.insert(sample.id).second) throw ContractError("duplicate sample ID");
    auto encoded = tokenizer.assemble(sample.input, task);
    result.push_back({sample.id, sample.family_id, std::move(encoded.token_ids),
                      std::move(encoded.segment_ids), sample.target, sample.answerable});
    start = end + 1;
  }
  std::sort(result.begin(), result.end(),
            [](const auto& left, const auto& right) { return left.id < right.id; });
  return result;
}

std::vector<size_t> epoch_order(size_t count, uint64_t seed, int64_t epoch) {
  if (epoch < 0) throw std::invalid_argument("negative epoch");
  std::vector<size_t> order(count);
  std::iota(order.begin(), order.end(), 0);
  // Same-toolchain reproducibility is recorded; the mixed seed separates epochs.
  std::seed_seq sequence{static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32),
                         static_cast<uint32_t>(epoch), 0x41524249U};
  std::mt19937_64 generator(sequence);
  std::shuffle(order.begin(), order.end(), generator);
  return order;
}

Batch make_batch(const std::vector<EncodedSample>& samples, const std::vector<size_t>& indices,
                 size_t begin, size_t batch_size) {
  if (begin >= indices.size() || batch_size == 0) throw std::invalid_argument("empty batch");
  const auto end = std::min(indices.size(), begin + batch_size);
  const auto count = end - begin;
  size_t length = 0;
  for (auto position = begin; position < end; ++position) {
    if (indices[position] >= samples.size()) throw std::out_of_range("sample index");
    length = std::max(length, samples[indices[position]].token_ids.size());
  }
  if (length == 0 || length > 256 || count > 64) throw std::invalid_argument("batch dimensions");
  auto ids = torch::zeros({static_cast<int64_t>(count), static_cast<int64_t>(length)}, torch::kInt64);
  auto mask = torch::zeros({static_cast<int64_t>(count), static_cast<int64_t>(length)}, torch::kBool);
  auto segments = torch::zeros_like(ids);
  auto targets = torch::full({static_cast<int64_t>(count)}, -1, torch::kInt64);
  auto answerable = torch::zeros({static_cast<int64_t>(count)}, torch::kBool);
  for (size_t row = 0; row < count; ++row) {
    const auto& sample = samples[indices[begin + row]];
    if (sample.token_ids.size() != sample.segment_ids.size() || sample.token_ids.empty())
      throw std::invalid_argument("invalid encoded sample");
    const auto width = static_cast<int64_t>(sample.token_ids.size());
    ids[static_cast<int64_t>(row)].slice(0, 0, width) = torch::tensor(sample.token_ids, torch::kInt64);
    segments[static_cast<int64_t>(row)].slice(0, 0, width) =
        torch::tensor(sample.segment_ids, torch::kInt64);
    mask[static_cast<int64_t>(row)].slice(0, 0, width) = true;
    targets[static_cast<int64_t>(row)] = sample.target;
    answerable[static_cast<int64_t>(row)] = sample.answerable;
  }
  return {ids, mask, segments, targets, answerable};
}

}  // namespace arbitrium
