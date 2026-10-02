#include "arbitrium/contracts.h"
#include "arbitrium/tokenizer.h"

#include <iostream>

int main(int argc, char** argv) {
  try {
    if (argc != 4) return 2;
    auto task = arbitrium::TaskSpec::from_json(arbitrium::parse_json(
        arbitrium::read_file(argv[1], 4 * 1024 * 1024), 4 * 1024 * 1024));
    arbitrium::Tokenizer tokenizer(argv[2], argv[3]);
    std::string line;
    while (std::getline(std::cin, line)) {
      auto request = arbitrium::DecisionRequest::from_json(arbitrium::parse_json(line), task);
      const auto value = tokenizer.assemble(request, task);
      std::cout << arbitrium::Json{{"token_ids", value.token_ids},
                                   {"segment_ids", value.segment_ids}}
                        .dump()
                << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 3;
  }
}
